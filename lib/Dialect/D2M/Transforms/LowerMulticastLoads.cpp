// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERMULTICASTLOADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Pattern to convert high-level multicast RemoteLoadOp to low-level form.
// High-level form uses mcast[dims] to specify which grid dimensions participate
// in multicast. Low-level form uses mcore[...] mshape[...] to specify explicit
// core start coordinates and multicast shape values.
class LowerMulticastLoadsRewriter : public OpRewritePattern<RemoteLoadOp> {
public:
  using OpRewritePattern<RemoteLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RemoteLoadOp op,
                                PatternRewriter &rewriter) const final {
    // Only match high-level multicast form
    if (!op.isHighLevelMcast()) {
      return failure();
    }

    // Get parent generic op to access grid shape
    auto genericOp = op->getParentOfType<GenericOp>();
    if (!genericOp) {
      return op.emitOpError("RemoteLoadOp must be inside a GenericOp");
    }

    ttcore::GridAttr grid = genericOp.getGrid();
    Location loc = op.getLoc();

    llvm::dbgs() << "Lowering multicast load op: \n" << op << "\n";

    // Extract mcast dimension indices from the high-level form.
    // The mcastDims are constant index values specifying which grid dimensions
    // should be multicast. The verifier guarantees these are constant indices.
    llvm::DenseSet<int64_t> mcastDimSet;
    for (Value dimValue : op.getMcastDims()) {
      auto constantOp = dimValue.getDefiningOp<arith::ConstantOp>();
      auto indexAttr = mlir::cast<IntegerAttr>(constantOp.getValue());
      mcastDimSet.insert(indexAttr.getInt());
    }

    // find operand index map
    auto memref = op.getMemref();
    auto operandGridShape = ttcore::getGridShape(memref);
    auto operandIndexingMap = genericOp.getIndexingMapForOperand(memref);

    // for each result in operand indexing map, check that the parallel
    // dimension is in mcastDimSet otherwise cannot construct a valid multicast
    bool implementAsUnicast = false;
    for (auto [idx, result] :
         llvm::enumerate(operandIndexingMap.getResults())) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(result)) {
        auto iterType = mlir::cast<ttcore::IteratorTypeAttr>(
            genericOp.getIteratorTypes()[dimExpr.getPosition()]);
        if (iterType.getValue() == ttcore::IteratorType::Parallel &&
            !mcastDimSet.contains(idx)) {
          implementAsUnicast = true;
        }
      } else {
        // if any expression isn't a dim expression, don't construct a multicast
        implementAsUnicast = true;
      }
    }

    // map parallel dims to the grid;
    auto outputIndexingMap = genericOp.getOutputIndexingMap();
    auto operandInvProjectedMap =
        inverseAndBroadcastProjectedPermutation(operandIndexingMap);
    auto outputInvProjectedMap =
        inverseAndBroadcastProjectedPermutation(outputIndexingMap);
    llvm::dbgs() << "grid: " << grid << "\n";
    llvm::dbgs() << "operandIndexingMap: " << operandIndexingMap << "\n";
    llvm::dbgs() << "operandInvProjectedMap: " << operandInvProjectedMap
                 << "\n";
    llvm::dbgs() << "outputInvProjectedMap: " << outputInvProjectedMap << "\n";
    // find intersection of outputInvMap and operandInvMap where results match
    for (auto [operandResult, outputResult] :
         llvm::zip(operandInvProjectedMap.getResults(),
                   outputInvProjectedMap.getResults())) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(operandResult)) {
        if (operandResult == outputResult) {
          // dim position here indexes compute grid dimension along multicast
          auto operandDimPosition = dimExpr.getPosition();
          auto multicastComputeGridDim = grid.getShape()[operandDimPosition];
          llvm::dbgs() << "    multicast group has size: "
                       << multicastComputeGridDim
                       << " for dim: " << operandDimPosition << "\n";
          // if multicast group is unit sized, just do unicast
          if (multicastComputeGridDim < 2) {
            implementAsUnicast = true;
          }
        }
      }
    }

    if (implementAsUnicast) {
      if (op.isExplicitCBForm()) {
        rewriter.replaceOpWithNewOp<RemoteLoadOp>(
            op, op.getCb(), op.getMemref(), op.getIndices());
      } else {
        rewriter.replaceOpWithNewOp<RemoteLoadOp>(
            op, op.getResult().getType(), op.getMemref(), op.getIndices());
      }
      return success();
    }

    // Build low-level multicast arguments.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));

    SmallVector<Value> mcastStartIndex;
    SmallVector<int64_t> mcastShapeInt64;
    mcastStartIndex.reserve(operandGridShape.size());
    mcastShapeInt64.reserve(operandGridShape.size());

    for (size_t dim = 0; dim < operandGridShape.size(); ++dim) {
      if (mcastDimSet.contains(static_cast<int64_t>(dim))) {
        Value coreIdx = rewriter.create<CoreIndexOp>(
            loc, static_cast<int64_t>(dim), grid.getMapping());
        mcastStartIndex.push_back(coreIdx);
        mcastShapeInt64.push_back(1);
      } else {
        mcastStartIndex.push_back(zero);
        mcastShapeInt64.push_back(operandGridShape[dim]);
      }
    }

    // Convert virtual multicast shape to physical shape if virtualization is
    // present.
    Value outputOperand = genericOp.getOutputs().front();
    auto outputShardLayout = mlir::cast<ttcore::ShardLayoutAttr>(
        ttcore::getDeviceLayout(outputOperand));
    if (!outputShardLayout.getCoreVirtualizationMap().isEmpty()) {
      // We project out the shard layout dims and results from the indexing
      // map before applying since we are only concerned with the grid
      // dimensions.
      auto coreVirtMap = outputShardLayout.getCoreVirtualizationMap();
      auto dimsToRemove = coreVirtMap.getNumResults() - mcastShapeInt64.size();
      llvm::SmallBitVector projectedDims(coreVirtMap.getNumDims());
      projectedDims.set(dimsToRemove, coreVirtMap.getNumDims());
      auto projectedMap = getProjectedMap(coreVirtMap, projectedDims);
      projectedMap = projectedMap.dropResults(projectedDims);
      mcastShapeInt64 = ttmlir::utils::evalShape(projectedMap, mcastShapeInt64);
    }

    // Convert int64_t mcast shape to Values.
    SmallVector<Value> mcastShape;
    mcastShape.reserve(mcastShapeInt64.size());
    for (int64_t dimSize : mcastShapeInt64) {
      mcastShape.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(dimSize)));
    }

    // Create replacement RemoteLoadOp with low-level multicast form.
    // The op can be in either CB form (no result) or result form.
    if (op.isExplicitCBForm()) {
      rewriter.replaceOpWithNewOp<RemoteLoadOp>(op, op.getCb(), op.getMemref(),
                                                op.getIndices(),
                                                mcastStartIndex, mcastShape);
    } else {
      rewriter.replaceOpWithNewOp<RemoteLoadOp>(op, op.getResult().getType(),
                                                op.getMemref(), op.getIndices(),
                                                mcastStartIndex, mcastShape);
    }

    return success();
  }
};

class D2MLowerMulticastLoads
    : public impl::D2MLowerMulticastLoadsBase<D2MLowerMulticastLoads> {
public:
  using impl::D2MLowerMulticastLoadsBase<
      D2MLowerMulticastLoads>::D2MLowerMulticastLoadsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMulticastLoadsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
