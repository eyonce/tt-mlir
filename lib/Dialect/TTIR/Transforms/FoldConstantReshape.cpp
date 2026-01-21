// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFOLDCONSTANTRESHAPE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if a reshape op's result is consumed only by elementwise binary ops
// (add, sub, mul, div, pow) or broadcast ops.
static bool isConsumedByElementwiseBinaryOrBroadcastOp(ReshapeOp reshapeOp) {
  for (Operation *user : reshapeOp->getUsers()) {
    if (!isa<ttir::AddOp, ttir::SubtractOp, ttir::MultiplyOp, ttir::DivOp,
             ttir::PowOp, ttir::BroadcastOp>(user)) {
      return false;
    }
  }
  return !reshapeOp->getUsers().empty();
}

// Trace through broadcast/typecast/reshape chains to find a splat constant.
// Returns the splat DenseElementsAttr if found, nullptr otherwise.
// This handles patterns like:
//   reshape(broadcast(reshape(constant))) where constant is a splat.
static DenseElementsAttr findSplatConstantThroughChain(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return nullptr;
  }

  // Direct constant case
  if (auto constantOp = dyn_cast<ConstantOp>(defOp)) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (denseAttr && denseAttr.isSplat()) {
      return denseAttr;
    }
    return nullptr;
  }

  // Trace through reshape - a splat reshaped is still a splat
  if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
    return findSplatConstantThroughChain(reshapeOp.getInput());
  }

  // Trace through broadcast - a splat broadcasted is still a splat
  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    return findSplatConstantThroughChain(broadcastOp.getInput());
  }

  // Trace through typecast - a splat typecast is still a splat (value changes
  // but splatness preserved)
  if (auto typecastOp = dyn_cast<TypecastOp>(defOp)) {
    return findSplatConstantThroughChain(typecastOp.getInput());
  }

  return nullptr;
}

// Pattern to fold reshape(splat_constant) into a new splat constant with the
// target shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<f32>}
//   %reshaped = ttir.reshape(%const) {shape = [1, 1]} -> tensor<1x1xf32>
//   %broadcast = ttir.broadcast(%reshaped) ...
//
// Transforms to:
//   %const = ttir.constant() {value = dense<2.0> : tensor<1x1xf32>}
//   %broadcast = ttir.broadcast(%const) ...
//
// Only handles splat constants consumed by elementwise binary ops or broadcast.
class FoldConstantReshapePattern : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Check if the input is a constant
    auto constantOp = reshapeOp.getInput().getDefiningOp<ConstantOp>();
    if (!constantOp) {
      return failure();
    }

    // Get the original constant value
    auto originalValue = constantOp.getValue();
    auto denseAttr = dyn_cast<DenseElementsAttr>(originalValue);
    if (!denseAttr) {
      return failure();
    }

    // Only handle splat constants
    if (!denseAttr.isSplat()) {
      return failure();
    }

    // Only fold if consumed by elementwise binary ops or broadcast
    if (!isConsumedByElementwiseBinaryOrBroadcastOp(reshapeOp)) {
      return failure();
    }

    // Get the target shape from the reshape op's result type
    auto reshapeResultType = cast<RankedTensorType>(reshapeOp.getType());
    auto targetShape = reshapeResultType.getShape();
    auto newType =
        RankedTensorType::get(targetShape, reshapeResultType.getElementType());

    // Create a new splat with the target shape
    DenseElementsAttr newDenseAttr;
    Type elementType = denseAttr.getElementType();
    if (isa<FloatType>(elementType)) {
      newDenseAttr =
          DenseElementsAttr::get(newType, denseAttr.getSplatValue<APFloat>());
    } else if (isa<IntegerType>(elementType)) {
      newDenseAttr =
          DenseElementsAttr::get(newType, denseAttr.getSplatValue<APInt>());
    } else {
      return failure();
    }

    // Create a new constant with the reshaped value
    auto newConstant =
        rewriter.create<ConstantOp>(constantOp.getLoc(), newType, newDenseAttr);

    // Replace the reshape op with the new constant
    rewriter.replaceOp(reshapeOp, newConstant.getResult());

    return success();
  }
};

// Pattern to fold reshape(broadcast/typecast/reshape chain from splat constant)
// into a new splat constant with the target shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<f32>}
//   %reshaped1 = ttir.reshape(%const) -> tensor<1x1x1xf32>
//   %broadcast = ttir.broadcast(%reshaped1) -> tensor<1x128x360xf32>
//   %reshaped2 = ttir.reshape(%broadcast) -> tensor<128x360xf32>
//   %result = ttir.pow(%activation, %reshaped2)
//
// Transforms to:
//   %const = ttir.constant() {value = dense<2.0> : tensor<128x360xf32>}
//   %result = ttir.pow(%activation, %const)
//
// Only handles splat constants consumed by elementwise binary ops or broadcast.
class FoldSplatConstantChainReshapePattern
    : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Skip if input is a direct constant (handled by
    // FoldConstantReshapePattern)
    if (reshapeOp.getInput().getDefiningOp<ConstantOp>()) {
      return failure();
    }

    // Only fold if consumed by elementwise binary ops or broadcast
    if (!isConsumedByElementwiseBinaryOrBroadcastOp(reshapeOp)) {
      return failure();
    }

    // Try to find a splat constant through the chain
    DenseElementsAttr splatAttr =
        findSplatConstantThroughChain(reshapeOp.getInput());
    if (!splatAttr) {
      return failure();
    }

    // Get the target shape and element type from the reshape op's result type
    auto reshapeResultType = cast<RankedTensorType>(reshapeOp.getType());
    auto targetShape = reshapeResultType.getShape();
    auto targetElementType = reshapeResultType.getElementType();

    // Get the splat value and create a new splat with the target type
    auto newType = RankedTensorType::get(targetShape, targetElementType);

    // Create a new splat constant with the target shape
    // We need to handle potential type conversion (e.g., if there was a
    // typecast in the chain)
    DenseElementsAttr newDenseAttr;
    Type splatElementType = splatAttr.getElementType();

    if (isa<FloatType>(splatElementType)) {
      APFloat splatValue = splatAttr.getSplatValue<APFloat>();

      if (splatElementType == targetElementType) {
        // Same element type - create splat with the same value
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else if (isa<FloatType>(targetElementType)) {
        // Float to float conversion (handles typecast in the chain)
        bool losesInfo = false;
        splatValue.convert(
            cast<FloatType>(targetElementType).getFloatSemantics(),
            APFloat::rmNearestTiesToEven, &losesInfo);
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else {
        // Unsupported conversion
        return failure();
      }
    } else if (isa<IntegerType>(splatElementType)) {
      APInt splatValue = splatAttr.getSplatValue<APInt>();

      if (splatElementType == targetElementType) {
        // Same element type - create splat with the same value
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else if (isa<IntegerType>(targetElementType)) {
        // Integer to integer conversion
        unsigned targetWidth = cast<IntegerType>(targetElementType).getWidth();
        if (targetWidth > splatValue.getBitWidth()) {
          splatValue = splatValue.sext(targetWidth);
        } else if (targetWidth < splatValue.getBitWidth()) {
          splatValue = splatValue.trunc(targetWidth);
        }
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else {
        // Unsupported conversion
        return failure();
      }
    } else {
      // Unsupported element type
      return failure();
    }

    // Create a new constant with the splat value at the target shape
    auto newConstant =
        rewriter.create<ConstantOp>(reshapeOp.getLoc(), newType, newDenseAttr);

    // Replace the reshape op with the new constant
    rewriter.replaceOp(reshapeOp, newConstant.getResult());

    return success();
  }
};

// Pattern to fold broadcast(splat_constant) into a new splat constant with the
// broadcasted shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<1x1xf32>}
//   %broadcast = ttir.broadcast(%const) -> tensor<128x360xf32>
//
// Transforms to:
//   %const = ttir.constant() {value = dense<2.0> : tensor<128x360xf32>}
//
// Only handles splat constants.
class FoldConstantBroadcastPattern : public OpRewritePattern<BroadcastOp> {
public:
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    // Check if the input is a constant
    auto constantOp = broadcastOp.getInput().getDefiningOp<ConstantOp>();
    if (!constantOp) {
      return failure();
    }

    // Get the original constant value
    auto originalValue = constantOp.getValue();
    auto denseAttr = dyn_cast<DenseElementsAttr>(originalValue);
    if (!denseAttr) {
      return failure();
    }

    // Only handle splat constants
    if (!denseAttr.isSplat()) {
      return failure();
    }

    // Get the target shape from the broadcast op's result type
    auto broadcastResultType = cast<RankedTensorType>(broadcastOp.getType());
    auto targetShape = broadcastResultType.getShape();
    auto newType = RankedTensorType::get(targetShape,
                                         broadcastResultType.getElementType());

    // Create a new splat with the target shape
    DenseElementsAttr newDenseAttr;
    Type elementType = denseAttr.getElementType();
    if (isa<FloatType>(elementType)) {
      newDenseAttr =
          DenseElementsAttr::get(newType, denseAttr.getSplatValue<APFloat>());
    } else if (isa<IntegerType>(elementType)) {
      newDenseAttr =
          DenseElementsAttr::get(newType, denseAttr.getSplatValue<APInt>());
    } else {
      return failure();
    }

    // Create a new constant with the broadcasted value
    auto newConstant =
        rewriter.create<ConstantOp>(constantOp.getLoc(), newType, newDenseAttr);

    // Replace the broadcast op with the new constant
    rewriter.replaceOp(broadcastOp, newConstant.getResult());

    return success();
  }
};

// Pattern to fold broadcast(broadcast/typecast/reshape chain from splat
// constant) into a new splat constant with the target shape.
class FoldSplatConstantChainBroadcastPattern
    : public OpRewritePattern<BroadcastOp> {
public:
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    // Skip if input is a direct constant (handled by
    // FoldConstantBroadcastPattern)
    if (broadcastOp.getInput().getDefiningOp<ConstantOp>()) {
      return failure();
    }

    // Try to find a splat constant through the chain
    DenseElementsAttr splatAttr =
        findSplatConstantThroughChain(broadcastOp.getInput());
    if (!splatAttr) {
      return failure();
    }

    // Get the target shape and element type from the broadcast op's result type
    auto broadcastResultType = cast<RankedTensorType>(broadcastOp.getType());
    auto targetShape = broadcastResultType.getShape();
    auto targetElementType = broadcastResultType.getElementType();

    // Get the splat value and create a new splat with the target type
    auto newType = RankedTensorType::get(targetShape, targetElementType);

    // Create a new splat constant with the target shape
    // We need to handle potential type conversion (e.g., if there was a
    // typecast in the chain)
    DenseElementsAttr newDenseAttr;
    Type splatElementType = splatAttr.getElementType();

    if (isa<FloatType>(splatElementType)) {
      APFloat splatValue = splatAttr.getSplatValue<APFloat>();

      if (splatElementType == targetElementType) {
        // Same element type - create splat with the same value
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else if (isa<FloatType>(targetElementType)) {
        // Float to float conversion (handles typecast in the chain)
        bool losesInfo = false;
        splatValue.convert(
            cast<FloatType>(targetElementType).getFloatSemantics(),
            APFloat::rmNearestTiesToEven, &losesInfo);
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else {
        // Unsupported conversion
        return failure();
      }
    } else if (isa<IntegerType>(splatElementType)) {
      APInt splatValue = splatAttr.getSplatValue<APInt>();

      if (splatElementType == targetElementType) {
        // Same element type - create splat with the same value
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else if (isa<IntegerType>(targetElementType)) {
        // Integer to integer conversion
        unsigned targetWidth = cast<IntegerType>(targetElementType).getWidth();
        if (targetWidth > splatValue.getBitWidth()) {
          splatValue = splatValue.sext(targetWidth);
        } else if (targetWidth < splatValue.getBitWidth()) {
          splatValue = splatValue.trunc(targetWidth);
        }
        newDenseAttr = DenseElementsAttr::get(newType, splatValue);
      } else {
        // Unsupported conversion
        return failure();
      }
    } else {
      // Unsupported element type
      return failure();
    }

    // Create a new constant with the splat value at the target shape
    auto newConstant = rewriter.create<ConstantOp>(broadcastOp.getLoc(),
                                                   newType, newDenseAttr);

    // Replace the broadcast op with the new constant
    rewriter.replaceOp(broadcastOp, newConstant.getResult());

    return success();
  }
};

class TTIRFoldConstantReshape
    : public impl::TTIRFoldConstantReshapeBase<TTIRFoldConstantReshape> {
public:
  using impl::TTIRFoldConstantReshapeBase<
      TTIRFoldConstantReshape>::TTIRFoldConstantReshapeBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConstantReshapePattern>(&getContext());
    patterns.add<FoldSplatConstantChainReshapePattern>(&getContext());
    patterns.add<FoldConstantBroadcastPattern>(&getContext());
    patterns.add<FoldSplatConstantChainBroadcastPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};

} // namespace
} // namespace mlir::tt::ttir
