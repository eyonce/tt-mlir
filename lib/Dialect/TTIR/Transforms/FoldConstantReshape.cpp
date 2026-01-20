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
// (add, sub, mul, div, pow).
static bool isConsumedByElementwiseBinaryOp(ReshapeOp reshapeOp) {
  for (Operation *user : reshapeOp->getUsers()) {
    if (!isa<ttir::AddOp, ttir::SubtractOp, ttir::MultiplyOp, ttir::DivOp,
             ttir::PowOp>(user)) {
      return false;
    }
  }
  return !reshapeOp->getUsers().empty();
}

// Pattern to fold reshape(constant) into a new constant with the target shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<32x1x2560xf32>}
//   %reshaped = ttir.reshape(%const) {shape = [32, 2560]} ->
//   tensor<32x2560xf32> %result = ttir.pow(%activation, %reshaped) :
//   tensor<32x2560xf32>
//
// Transforms to:
//   %const = ttir.constant() {value = dense<2.0> : tensor<32x2560xf32>}
//   %result = ttir.pow(%activation, %const) : tensor<32x2560xf32>
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

    // Check if the reshape result is consumed by elementwise binary ops
    if (!isConsumedByElementwiseBinaryOp(reshapeOp)) {
      return failure();
    }

    // Get the original constant value
    auto originalValue = constantOp.getValue();
    auto denseAttr = dyn_cast<DenseElementsAttr>(originalValue);
    if (!denseAttr) {
      return failure();
    }

    // Get the target shape from the reshape op's result type
    auto reshapeResultType = cast<RankedTensorType>(reshapeOp.getType());
    auto targetShape = reshapeResultType.getShape();

    // Create a new DenseElementsAttr with the target shape
    auto newType =
        RankedTensorType::get(targetShape, reshapeResultType.getElementType());
    auto newDenseAttr = denseAttr.reshape(newType);

    // Create a new constant with the reshaped value
    auto newConstant =
        rewriter.create<ConstantOp>(constantOp.getLoc(), newType, newDenseAttr);

    // Replace the reshape op with the new constant
    rewriter.replaceOp(reshapeOp, newConstant.getResult());

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
