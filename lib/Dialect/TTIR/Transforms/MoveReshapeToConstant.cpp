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
#define GEN_PASS_DEF_TTIRMOVERESHAPETOCONSTANT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if a value traces back to a constant op.
static bool isFromConstant(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return false;
  }

  // Direct constant
  if (isa<ttir::ConstantOp>(defOp)) {
    return true;
  }

  // Reshape/broadcast/typecast of a constant is still "from constant"
  if (isa<ttir::ReshapeOp, ttir::BroadcastOp, ttir::TypecastOp>(defOp)) {
    return isFromConstant(defOp->getOperand(0));
  }

  return false;
}

// Pattern to move reshapes from activation paths to constant paths in
// elementwise binary operations.
//
// Matches patterns like:
//   %const = ttir.constant() : tensor<32x1x2560xf32>
//   %act = ... : tensor<32x2560xf32>
//   %reshaped = ttir.reshape(%act) : tensor<32x2560xf32> ->
//   tensor<32x1x2560xf32> %result = ttir.pow(%reshaped, %const) :
//   tensor<32x1x2560xf32>
//
// Transforms to:
//   %const = ttir.constant() : tensor<32x1x2560xf32>
//   %const_reshaped = ttir.reshape(%const) : tensor<32x1x2560xf32> ->
//   tensor<32x2560xf32> %act = ... : tensor<32x2560xf32> %result =
//   ttir.pow(%act, %const_reshaped) : tensor<32x2560xf32>
class MoveReshapeToConstantPattern
    : public OpInterfaceRewritePattern<ElementwiseBinary> {
public:
  using OpInterfaceRewritePattern<ElementwiseBinary>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ElementwiseBinary op,
                                PatternRewriter &rewriter) const override {
    // Get operands
    auto operands = op->getOperands();
    if (operands.size() != 2) {
      return failure();
    }

    Value lhs = operands[0];
    Value rhs = operands[1];

    // Find which operand is a reshape and which traces to a constant
    ReshapeOp reshapeOp = nullptr;
    Value constOperand = nullptr;
    size_t reshapeOperandIdx = 0;

    auto lhsReshape = lhs.getDefiningOp<ReshapeOp>();
    auto rhsReshape = rhs.getDefiningOp<ReshapeOp>();

    // Case 1: LHS is reshape, RHS is from constant
    if (lhsReshape && !isFromConstant(lhs) && isFromConstant(rhs)) {
      reshapeOp = lhsReshape;
      constOperand = rhs;
      reshapeOperandIdx = 0;
    }
    // Case 2: RHS is reshape, LHS is from constant
    else if (rhsReshape && !isFromConstant(rhs) && isFromConstant(lhs)) {
      reshapeOp = rhsReshape;
      constOperand = lhs;
      reshapeOperandIdx = 1;
    } else {
      return failure();
    }

    // The reshape must have a single use (the elementwise op)
    if (!reshapeOp->hasOneUse()) {
      return failure();
    }

    // Get the pre-reshape shape (activation's original shape)
    auto preReshapeType =
        cast<RankedTensorType>(reshapeOp.getInput().getType());
    auto postReshapeType = cast<RankedTensorType>(reshapeOp.getType());

    // The constant operand should have the same shape as the post-reshape
    // (that's why the reshape was added)
    auto constType = cast<RankedTensorType>(constOperand.getType());
    if (constType.getShape() != postReshapeType.getShape()) {
      return failure();
    }

    // Create inverse reshape for the constant: from post-reshape shape to
    // pre-reshape shape
    SmallVector<int32_t> newShape;
    for (int64_t dim : preReshapeType.getShape()) {
      newShape.push_back(static_cast<int32_t>(dim));
    }

    auto newConstType = RankedTensorType::get(preReshapeType.getShape(),
                                              constType.getElementType());

    auto constReshape = rewriter.create<ReshapeOp>(
        constOperand.getLoc(), newConstType, constOperand,
        rewriter.getI32ArrayAttr(newShape));

    // Create the new elementwise op with the original activation and reshaped
    // constant
    SmallVector<Value> newOperands;
    if (reshapeOperandIdx == 0) {
      newOperands.push_back(reshapeOp.getInput());
      newOperands.push_back(constReshape.getResult());
    } else {
      newOperands.push_back(constReshape.getResult());
      newOperands.push_back(reshapeOp.getInput());
    }

    // The result type should match the pre-reshape type
    auto newResultType = RankedTensorType::get(
        preReshapeType.getShape(),
        cast<RankedTensorType>(op->getResult(0).getType()).getElementType());

    Operation *newOp = rewriter.create(
        op->getLoc(), rewriter.getStringAttr(op->getName().getStringRef()),
        newOperands, newResultType, op->getAttrs());

    // Replace the original op result with the new op result
    // We need a reshape to match the original output shape for downstream users
    auto outputReshape = rewriter.create<ReshapeOp>(
        op->getLoc(), op->getResult(0).getType(), newOp->getResult(0),
        reshapeOp.getShapeAttr());

    rewriter.replaceOp(op, outputReshape.getResult());

    return success();
  }
};

class TTIRMoveReshapeToConstant
    : public impl::TTIRMoveReshapeToConstantBase<TTIRMoveReshapeToConstant> {
public:
  using impl::TTIRMoveReshapeToConstantBase<
      TTIRMoveReshapeToConstant>::TTIRMoveReshapeToConstantBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MoveReshapeToConstantPattern>(&getContext());
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
