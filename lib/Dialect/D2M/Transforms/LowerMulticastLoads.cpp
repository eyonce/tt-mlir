// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"

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
    ArrayRef<int64_t> gridShape = grid.getShape();
    Location loc = op.getLoc();

    // Extract mcast dimension indices from the high-level form.
    // The mcastDims are constant index values specifying which grid dimensions
    // should be multicast. The verifier guarantees these are constant indices.
    llvm::DenseSet<int64_t> mcastDimSet;
    for (Value dimValue : op.getMcastDims()) {
      auto constantOp = dimValue.getDefiningOp<arith::ConstantOp>();
      auto indexAttr = mlir::cast<IntegerAttr>(constantOp.getValue());
      mcastDimSet.insert(indexAttr.getInt());
    }

    // If all multicast dimensions have grid size 1, strip multicast and lower
    // to a regular unicast remote load.
    bool isUnicast = llvm::all_of(
        mcastDimSet, [&](int64_t dim) { return gridShape[dim] == 1; });
    if (isUnicast) {
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
    // For each grid dimension:
    // - If dim is in mcastDims (multicast dimension):
    //   - mcastStartIndex[dim] = 1 (sender at core 0, multicast starts at 1)
    //   - mcastShape[dim] = gridShape[dim] - 1
    // - If dim is NOT in mcastDims (parallel dimension):
    //   - mcastStartIndex[dim] = core_index(dim)
    //   - mcastShape[dim] = 1
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                   rewriter.getIndexAttr(1));

    SmallVector<Value> mcastStartIndex;
    SmallVector<Value> mcastShape;
    mcastStartIndex.reserve(gridShape.size());
    mcastShape.reserve(gridShape.size());

    for (size_t dim = 0; dim < gridShape.size(); ++dim) {
      if (mcastDimSet.contains(static_cast<int64_t>(dim))) {
        // Multicast dimension: sender at core 0, multicast to all other cores
        int64_t numDests = gridShape[dim] - 1;
        Value gridDimMinusOne = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexType(), rewriter.getIndexAttr(numDests));
        mcastStartIndex.push_back(one);
        mcastShape.push_back(gridDimMinusOne);
      } else {
        // Parallel dimension: mcast to self only
        Value coreIdx =
            rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(dim));
        mcastStartIndex.push_back(coreIdx);
        mcastShape.push_back(one);
      }
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
