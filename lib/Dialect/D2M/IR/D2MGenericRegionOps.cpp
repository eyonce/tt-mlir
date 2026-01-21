// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/APInt.h"
#include <limits>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

// Template helper for bufferizing CB ops that take a CB operand but no result
// (PushOp, PopOp). These ops unwrap to_tensor by creating to_buffer.
template <typename OpTy>
static mlir::LogicalResult
bufferizeCBOp(OpTy op, mlir::RewriterBase &rewriter,
              const mlir::bufferization::BufferizationOptions &options) {
  auto cbBufferType =
      mlir::cast<bufferization::TensorLikeType>(op.getCbType())
          .getBufferType(options, [&]() { return op.emitOpError(); });
  assert(succeeded(cbBufferType));
  auto toBuffer = rewriter.create<bufferization::ToBufferOp>(
      op.getLoc(), *cbBufferType, op.getCb());
  mlir::bufferization::replaceOpWithNewBufferizedOp<OpTy>(rewriter, op,
                                                          toBuffer.getResult());
  return mlir::success();
}

void AcquireDstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "dst");
}

// Helper to extract element type from ranked tensor or memref
static Type getElemType(Type ty) {
  if (auto rt = dyn_cast<RankedTensorType>(ty)) {
    return rt.getElementType();
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    return mr.getElementType();
  }
  return ty;
}

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

//===----------------------------------------------------------------------===//
// DMA Operations
//===----------------------------------------------------------------------===//

// DMA base verification and helpers
mlir::LogicalResult DMAOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  bool skipChecksForTensorType = mlir::isa<RankedTensorType>(srcType) ||
                                 mlir::isa<RankedTensorType>(dstType);

  if (!skipChecksForTensorType &&
      srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("operands to DMA must have the same element type");
  }

  if (isSrcRemote() && isDstRemote()) {
    return emitOpError("cannot have both src and dst remote");
  }

  if (isDstRemote() && isMcast()) {
    return emitOpError("cannot mcast to remote dst");
  }

  if (getSrcAffineMap() && !getSrcIndices().empty()) {
    return emitOpError("cannot have both src affine map and indices");
  }

  if (getDstAffineMap() && !getDstIndices().empty()) {
    return emitOpError("cannot have both dst affine map and indices");
  }

  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast start index requires mcast shape");
  }

  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast shape requires mcast start index");
  }

  if (skipChecksForTensorType) {
    return success();
  }

  int64_t srcIndices = getSrcAffineMap() ? getSrcAffineMap()->getNumResults()
                                         : getSrcIndices().size();
  int64_t dstIndices = getDstAffineMap() ? getDstAffineMap()->getNumResults()
                                         : getDstIndices().size();

  if (srcIndices > srcType.getRank()) {
    return emitOpError("invalid number of src indices, expected less than ")
           << srcType.getRank();
  }

  if (dstIndices > dstType.getRank()) {
    return emitOpError("invalid number of dst indices, expected less than ")
           << dstType.getRank();
  }

  if ((srcType.getRank() - srcIndices) != (dstType.getRank() - dstIndices)) {
    return emitOpError("operands must have the same post-index rank");
  }

  if (!std::equal(srcType.getShape().begin() + srcIndices,
                  srcType.getShape().end(),
                  dstType.getShape().begin() + dstIndices)) {
    return emitOpError("operands must have the same post-index shape");
  }

  if (getSrcAffineMap() && !isSrcRemote()) {
    return emitOpError("if src affine map is provided, src must be remote");
  }

  if (getDstAffineMap() && !isDstRemote()) {
    return emitOpError("if dst affine map is provided, dst must be remote");
  }

  return success();
}

int64_t DMAOp::getNumElems() {
  if (getOptNumElems()) {
    return *getOptNumElems();
  }
  ArrayRef<int64_t> txShape =
      getSrcMemRefType().getShape().drop_front(getSrcIndices().size());
  return ttmlir::utils::volume(txShape);
}

size_t DMAOp::getSizeBytes() {
  auto elementSizeBytes =
      ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
  return getNumElems() * elementSizeBytes;
}

void DMAOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

bool DMAOp::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                             const mlir::bufferization::AnalysisState &) {
  // Return true to avoid forcing out of place bufferization.
  return true;
}

// Comprehensive verifiers matching D2M
::mlir::LogicalResult DMAWriteOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  auto isRemote = [&](auto operand) {
    return ttcore::hasDeviceLayout(operand);
  };

  if (isRemote(getSrc())) {
    return emitOpError("For DMAWrite, src must be local");
  }

  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("Operands to DMAWrite must have the same element type");
  }

  if (isDstRemote() && isMcast()) {
    return emitOpError("cannot mcast to remote dst");
  }

  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast shape defined but mcast start index is not");
  }

  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast start index defined but mcast shape is not");
  }

  constexpr int64_t kExpectedIndicesRemote = 3;
  constexpr int64_t kExpectedIndicesLocal = 1;

  int64_t numDstIndices = getDstIndices().size();
  int64_t numSrcIndices = getSrcIndices().size();

  if (isDstRemote()) {
    if (numDstIndices != kExpectedIndicesRemote) {
      return emitOpError("Must have 3 dst indices for remote dst operand");
    }
  } else {
    if (numDstIndices != kExpectedIndicesLocal) {
      return emitOpError("Must have 1 dst index for local dst operand");
    }
  }

  if (numSrcIndices != kExpectedIndicesLocal) {
    return emitOpError("Must have 1 src index for local src operand");
  }
  return success();
}

::mlir::LogicalResult DMAReadOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());
  auto isRemote = [&](auto operand) {
    return ttcore::hasDeviceLayout(operand);
  };
  auto isLocal = [&](auto operand) { return !isRemote(operand); };
  if (!(isRemote(getSrc()) && isLocal(getDst()))) {
    return emitOpError("For DMARead, src must be remote and dst must be local");
  }
  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("Operands to DMAWrite must have the same element type");
  }
  constexpr int64_t kExpectedIndicesRemote = 3;
  constexpr int64_t kExpectedIndicesLocal = 1;
  int64_t numDstIndices = getDstIndices().size();
  int64_t numSrcIndices = getSrcIndices().size();
  if (numSrcIndices != kExpectedIndicesRemote) {
    return emitOpError("Must have 3 src indices for remote src operand");
  }
  if (numDstIndices != kExpectedIndicesLocal) {
    return emitOpError("Must have 1 dst index for local dst operand");
  }
  return success();
}

void DMAReadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void DMAWriteOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void DMAOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                           mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
}

void DMAReadOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
}

void DMAWriteOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
}

bool DMAOp::bufferizesToMemoryRead(mlir::OpOperand &operand,
                                   const mlir::bufferization::AnalysisState &) {
  return operand.get() == getSrc();
}

bool DMAOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getDst();
}

mlir::bufferization::AliasingValueList
DMAOp::getAliasingValues(mlir::OpOperand &,
                         const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
DMAOp::getBufferType(mlir::Value value,
                     const mlir::bufferization::BufferizationOptions &,
                     const mlir::bufferization::BufferizationState &,
                     ::llvm::SmallVector<mlir::Value> &) {
  auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  return mlir::tt::ttcore::getBufferType(rankedTensorType, /*isView=*/true);
}

mlir::LogicalResult
DMAOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options,
                 mlir::bufferization::BufferizationState &state) {
  mlir::FailureOr<Value> src =
      mlir::bufferization::getBuffer(rewriter, getSrc(), options, state);
  // NOLINTNEXTLINE
  if (failed(src)) {
    return src;
  }

  mlir::FailureOr<Value> dst =
      mlir::bufferization::getBuffer(rewriter, getDst(), options, state);
  // NOLINTNEXTLINE
  if (failed(dst)) {
    return dst;
  }

  ::llvm::SmallVector<mlir::Value> invocationStack;
  // NOLINTNEXTLINE
  mlir::bufferization::replaceOpWithNewBufferizedOp<mlir::tt::d2m::DMAOp>(
      rewriter, *this, getResult().getType(), *src, getSrcAffineMapAttr(),
      getSrcIndices(), *dst, getDstAffineMapAttr(), getDstIndices(),
      getOptNumElemsAttr(), getMcastStartIndex(), getMcastShape());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Remote Load/Store Operations
//===----------------------------------------------------------------------===//

::mlir::LogicalResult RemoteLoadOp::verify() {
  auto shapedType = getShapedType();
  bool hasCbOperand = static_cast<bool>(getCb());
  bool hasResultValue = static_cast<bool>(getResult());

  // Verify XOR constraint: exactly one of cb or result must be present
  if (hasCbOperand == hasResultValue) {
    if (hasCbOperand) {
      return emitOpError(
          "cannot have both circular buffer and result; exactly one must be "
          "present");
    }
    return emitOpError(
        "must have either circular buffer or result; exactly one must be "
        "present");
  }

  // Verify that tensor parameters are not allowed in explicit CB form
  if (isExplicitCBForm()) {
    if (mlir::isa<RankedTensorType>(getMemref().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; memref "
          "operand must be a memref type");
    }
    if (hasResultValue && mlir::isa<RankedTensorType>(getResult().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; result must "
          "be a memref type");
    }
  }

  // Verify that the memref/tensor is remote (has device layout)
  if (!ttcore::hasDeviceLayout(getMemref())) {
    return emitOpError("memref/tensor must be remote (have a device layout)");
  }

  // Verify memref/tensor rank is even (grid + shard dimensions)
  if (shapedType.getRank() % 2 != 0) {
    return emitOpError("memref/tensor rank must be even for device shape (grid "
                       "+ shard dimensions)");
  }

  // Verify indices count matches grid dimensions (first N/2 dimensions)
  int64_t gridRank = shapedType.getRank() / 2;
  if (static_cast<int64_t>(getIndices().size()) != gridRank) {
    return emitOpError("number of indices must equal grid rank (N/2 where N is "
                       "memref/tensor rank), got ")
           << getIndices().size() << " indices but expected " << gridRank;
  }

  // Verify multicast parameters: both must be provided or neither
  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast start index requires mcast shape");
  }
  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast shape requires mcast start index");
  }

  // Verify mutual exclusivity between low-level and high-level multicast forms
  if (!getMcastShape().empty() && !getMcastDims().empty()) {
    return emitOpError(
        "cannot use both low-level multicast (mcore/mshape) and high-level "
        "multicast (mcast dims); they are mutually exclusive");
  }

  // Verify high-level mcast dimensions are constant indices
  SmallVector<int64_t> mcastDimIndices;
  for (Value dimValue : getMcastDims()) {
    auto constantOp = dimValue.getDefiningOp<arith::ConstantOp>();
    if (!constantOp) {
      return emitOpError("mcast dimension must be a constant index");
    }
    auto indexAttr = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!indexAttr) {
      return emitOpError("mcast dimension must be an integer attribute");
    }
    mcastDimIndices.push_back(indexAttr.getInt());
  }

  // Verify that memref references a generic op operand when inside a generic
  if (auto genericOp = getOperation()->getParentOfType<GenericOp>()) {
    Value memrefOperand = getMemref();
    std::optional<unsigned> operandIndex;
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOperands())) {
      if (operand == memrefOperand) {
        operandIndex = idx;
        break;
      }
    }
    if (!operandIndex) {
      return emitOpError(
          "memref operand must reference one of the parent generic op's "
          "operands directly");
    }

    // Verify mcast dimensions are parallel iterator type
    if (!mcastDimIndices.empty()) {
      AffineMap indexingMap = genericOp.getIndexingMap(*operandIndex);
      ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

      for (int64_t gridDim : mcastDimIndices) {
        if (gridDim < 0 ||
            gridDim >= static_cast<int64_t>(indexingMap.getNumResults())) {
          return emitOpError("mcast dimension index ")
                 << gridDim << " is out of bounds for grid rank "
                 << indexingMap.getNumResults();
        }

        AffineExpr expr = indexingMap.getResult(gridDim);
        if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
          int64_t iterDimPos = dimExpr.getPosition();
          auto iterType =
              mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[iterDimPos]);
          if (iterType.getValue() != ttcore::IteratorType::Parallel) {
            return emitOpError("mcast dimension index ")
                   << gridDim
                   << " must correspond to a parallel iterator type, but "
                      "found reduction";
          }
        }
      }
    }
  }

  // CB-specific verification (only when CB is present)
  if (hasCbOperand) {
    auto cbType = mlir::cast<CBType>(getCb().getType());

    // Verify CB type matches shard shape
    auto deviceLayout = ttcore::getDeviceLayout(getMemref());
    if (!deviceLayout) {
      return emitOpError("failed to get device layout from memref/tensor");
    }

    auto shardShape = deviceLayout.getShardShape(shapedType);
    auto cbUnderlyingType = cbType.getUnderlying();

    // Verify CB underlying shape matches shard shape
    if (cbUnderlyingType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError(
                 "circular buffer underlying rank must match shard shape "
                 "rank, got ")
             << cbUnderlyingType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != mlir::ShapedType::kDynamic &&
          cbUnderlyingType.getDimSize(i) != mlir::ShapedType::kDynamic &&
          shardShape[i] != cbUnderlyingType.getDimSize(i)) {
        return emitOpError("circular buffer underlying shape must match shard "
                           "shape at dimension ")
               << i << ", got " << cbUnderlyingType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    Type cbElementType = cbUnderlyingType.getElementType();
    Type shapedElementType = shapedType.getElementType();
    if (cbElementType != shapedElementType) {
      return emitOpError(
          "circular buffer element type must match memref/tensor element type");
    }
  }

  return mlir::success();
}

::mlir::LogicalResult RemoteStoreOp::verify() {
  auto shapedType = getShapedType();
  bool hasCbOperand = static_cast<bool>(getCb());
  bool hasLocalBufferOperand = static_cast<bool>(getLocalBuffer());
  bool hasResultValue = static_cast<bool>(getResult());

  // Verify XOR constraint: exactly one of localBuffer or cb must be present
  if (hasCbOperand == hasLocalBufferOperand) {
    if (hasCbOperand) {
      return emitOpError(
          "cannot have both circular buffer and local buffer; exactly one must "
          "be present");
    }
    return emitOpError(
        "must have either circular buffer or local buffer; exactly one must be "
        "present");
  }

  // Verify result presence constraints (mirror RemoteLoadOp)
  // Explicit CB form: CB present, localBuffer absent, result must NOT be
  // present Implicit form: localBuffer present, CB absent, result MUST be
  // present
  if (hasCbOperand && hasResultValue) {
    return emitOpError(
        "explicit CB form cannot have a result; result must only be present in "
        "implicit form");
  }
  if (hasLocalBufferOperand && !hasResultValue) {
    return emitOpError("implicit form (with localBuffer) must have a result");
  }

  // Verify that tensor parameters are not allowed in explicit CB form
  if (isExplicitCBForm()) {
    if (mlir::isa<RankedTensorType>(getMemref().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; memref "
          "operand must be a memref type");
    }
    if (hasLocalBufferOperand &&
        mlir::isa<RankedTensorType>(getLocalBuffer().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; localBuffer "
          "operand must be a memref type");
    }
  }

  // Verify that the memref/tensor is remote (has device layout)
  if (!ttcore::hasDeviceLayout(getMemref())) {
    return emitOpError("memref/tensor must be remote (have a device layout)");
  }

  // Verify memref/tensor rank is even (grid + shard dimensions)
  if (shapedType.getRank() % 2 != 0) {
    return emitOpError("memref/tensor rank must be even for device shape (grid "
                       "+ shard dimensions)");
  }

  // Verify indices count matches grid dimensions (first N/2 dimensions)
  int64_t gridRank = shapedType.getRank() / 2;
  if (static_cast<int64_t>(getIndices().size()) != gridRank) {
    return emitOpError("number of indices must equal grid rank (N/2 where N is "
                       "memref/tensor rank), got ")
           << getIndices().size() << " indices but expected " << gridRank;
  }

  // Verify that memref references a generic op operand when inside a generic
  if (auto genericOp = getOperation()->getParentOfType<GenericOp>()) {
    Value memrefOperand = getMemref();
    bool foundInOperands = false;
    for (Value operand : genericOp.getOperands()) {
      if (operand == memrefOperand) {
        foundInOperands = true;
        break;
      }
    }
    if (!foundInOperands) {
      return emitOpError(
          "memref operand must reference one of the parent generic op's "
          "operands directly");
    }
  }

  // Get device layout for shape verification
  auto deviceLayout = ttcore::getDeviceLayout(getMemref());
  if (!deviceLayout) {
    return emitOpError("failed to get device layout from memref/tensor");
  }
  auto shardShape = deviceLayout.getShardShape(shapedType);

  // CB-specific verification (only when CB is present)
  if (hasCbOperand) {
    auto cbType = mlir::cast<CBType>(getCb().getType());
    auto cbUnderlyingType = cbType.getUnderlying();

    // Verify CB underlying shape matches shard shape
    if (cbUnderlyingType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError(
                 "circular buffer underlying rank must match shard shape "
                 "rank, got ")
             << cbUnderlyingType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != mlir::ShapedType::kDynamic &&
          cbUnderlyingType.getDimSize(i) != mlir::ShapedType::kDynamic &&
          shardShape[i] != cbUnderlyingType.getDimSize(i)) {
        return emitOpError("circular buffer underlying shape must match shard "
                           "shape at dimension ")
               << i << ", got " << cbUnderlyingType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    Type cbElementType = cbUnderlyingType.getElementType();
    Type shapedElementType = shapedType.getElementType();
    if (cbElementType != shapedElementType) {
      return emitOpError(
          "circular buffer element type must match memref/tensor element type");
    }
  }

  // Local buffer-specific verification (only when localBuffer is present)
  if (hasLocalBufferOperand) {
    auto localBufferType = mlir::cast<ShapedType>(getLocalBuffer().getType());

    // Verify local buffer shape matches shard shape
    if (localBufferType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError("local buffer rank must match shard shape rank, got ")
             << localBufferType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != mlir::ShapedType::kDynamic &&
          localBufferType.getDimSize(i) != mlir::ShapedType::kDynamic &&
          shardShape[i] != localBufferType.getDimSize(i)) {
        return emitOpError(
                   "local buffer shape must match shard shape at dimension ")
               << i << ", got " << localBufferType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    Type localBufferElementType = localBufferType.getElementType();
    Type shapedElementType = shapedType.getElementType();
    if (localBufferElementType != shapedElementType) {
      return emitOpError(
          "local buffer element type must match memref/tensor element type");
    }
  }

  // Verify result type matches memref type if result is present and in tensor
  // mode
  if (hasResultValue) {
    Type memrefType = getMemref().getType();

    // Only verify result type in tensor mode
    if (mlir::isa<RankedTensorType>(memrefType)) {
      Type resultType = getResult().getType();
      // Result should match the destination memref type
      if (resultType != memrefType) {
        return emitOpError("result type must match memref type");
      }
    } else {
      // In memref mode, result should be unused
      if (!getResult().use_empty()) {
        return emitOpError(
            "result must be unused when memref operand has memref type");
      }
    }
  }

  return mlir::success();
}

void RemoteLoadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Load operations don't have results anymore (they load into CB)
}

//===----------------------------------------------------------------------===//
// RemoteLoadOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool RemoteLoadOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Only result-form (no CB) operations should exist during bufferization
  // The memref operand is read from
  return operand.get() == getMemref();
}

bool RemoteLoadOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  // Only result-form (no CB) operations should exist during bufferization
  return false;
}

mlir::bufferization::AliasingValueList
RemoteLoadOp::getAliasingValues(mlir::OpOperand &,
                                const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
RemoteLoadOp::getBufferType(mlir::Value value,
                            const mlir::bufferization::BufferizationOptions &,
                            const mlir::bufferization::BufferizationState &,
                            ::llvm::SmallVector<mlir::Value> &) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return mlir::failure();
  }
  if (value == getMemref()) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  return mlir::failure();
}

mlir::LogicalResult RemoteLoadOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return emitOpError(
        "RemoteLoadOp with CB should not exist during bufferization");
  }

  // Result-only mode: no CB, just the result
  Value result = getResult();
  if (!result) {
    return emitOpError("Expected result when CB is not present");
  }

  // Bufferize the memref/tensor operand
  mlir::FailureOr<Value> memrefBuffer =
      mlir::bufferization::getBuffer(rewriter, getMemref(), options, state);
  // NOLINTNEXTLINE
  if (failed(memrefBuffer)) {
    return memrefBuffer;
  }

  // Convert result type to memref type
  Type resultBufferType =
      ttcore::getBufferType(result.getType(), /*isView=*/false);

  // RemoteLoadOp always loads into L1 memory, so ensure the result type has
  // L1 memory space. If the result type is a memref without memory space,
  // add the L1 memory space attribute.
  if (auto memrefType = mlir::dyn_cast<MemRefType>(resultBufferType)) {
    if (!memrefType.getMemorySpace()) {
      auto l1Attr = ttcore::MemorySpaceAttr::get(getContext(),
                                                 ttcore::MemorySpace::DeviceL1);
      resultBufferType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          memrefType.getLayout(), l1Attr);
    }
  }

  // Create a new RemoteLoadOp with bufferized operands (no CB, with result)
  // Preserve the multicast form - either high-level (mcastDims) or low-level
  // (mcastStartIndex/mcastShape)
  if (isHighLevelMcast()) {
    // High-level mcast form: use mcastDims builder
    // NOLINTNEXTLINE
    mlir::bufferization::replaceOpWithNewBufferizedOp<RemoteLoadOp>(
        rewriter, *this, resultBufferType, *memrefBuffer, getIndices(),
        getMcastDims());
  } else {
    // Low-level mcast form or no mcast: use mcastStartIndex/mcastShape builder
    // NOLINTNEXTLINE
    mlir::bufferization::replaceOpWithNewBufferizedOp<RemoteLoadOp>(
        rewriter, *this, resultBufferType, *memrefBuffer, getIndices(),
        getMcastStartIndex(), getMcastShape());
  }

  return mlir::success();
}

bool RemoteLoadOp::hasTensorSemantics() {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return false;
  }

  // Check if the memref operand is a tensor (needs bufferization)
  bool memrefIsTensor = mlir::isa<RankedTensorType>(getMemref().getType());
  // Check if the result is a tensor (needs bufferization)
  Value result = getResult();
  bool resultIsTensor = result && mlir::isa<RankedTensorType>(result.getType());
  return memrefIsTensor || resultIsTensor;
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool RemoteStoreOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Only localBuffer-form (no CB) operations should exist during bufferization
  // The localBuffer operand is read from
  return operand.get() == getLocalBuffer();
}

bool RemoteStoreOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // The memref operand is written to
  return operand.get() == getMemref();
}

mlir::bufferization::AliasingValueList
RemoteStoreOp::getAliasingValues(mlir::OpOperand &,
                                 const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
RemoteStoreOp::getBufferType(mlir::Value value,
                             const mlir::bufferization::BufferizationOptions &,
                             const mlir::bufferization::BufferizationState &,
                             ::llvm::SmallVector<mlir::Value> &) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return mlir::failure();
  }

  Value localBuffer = getLocalBuffer();
  if (!localBuffer) {
    return mlir::failure();
  }

  Value result = getResult();
  if (result && value == result) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }

  if (value == localBuffer) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  if (value == getMemref()) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  return mlir::failure();
}

mlir::LogicalResult RemoteStoreOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return emitOpError(
        "RemoteStoreOp with CB should not exist during bufferization");
  }

  // Implicit form: localBuffer mode
  Value localBuffer = getLocalBuffer();
  if (!localBuffer) {
    return emitOpError("Expected localBuffer when CB is not present");
  }

  // Bufferize the memref/tensor operand
  mlir::FailureOr<Value> memrefBuffer =
      mlir::bufferization::getBuffer(rewriter, getMemref(), options, state);
  // NOLINTNEXTLINE
  if (failed(memrefBuffer)) {
    return memrefBuffer;
  }

  // Bufferize the localBuffer operand (only if it's a tensor)
  Value localBufferBufferized = localBuffer;
  if (mlir::isa<RankedTensorType>(localBuffer.getType())) {
    mlir::FailureOr<Value> localBufferMaybe =
        mlir::bufferization::getBuffer(rewriter, localBuffer, options, state);
    // NOLINTNEXTLINE
    if (failed(localBufferMaybe)) {
      return localBufferMaybe;
    }
    localBufferBufferized = *localBufferMaybe;
  }

  // Convert result type to memref type
  // In implicit form (localBuffer present), result is always required
  Value result = getResult();
  if (!result) {
    return emitOpError("Expected result in implicit form during bufferization");
  }

  Type resultBufferType =
      ttcore::getBufferType(result.getType(), /*isView=*/false);

  // Create a new RemoteStoreOp with bufferized operands and result
  // NOLINTNEXTLINE
  mlir::bufferization::replaceOpWithNewBufferizedOp<RemoteStoreOp>(
      rewriter, *this, resultBufferType, *memrefBuffer, getIndices(),
      localBufferBufferized, /*cb=*/Value{});

  return mlir::success();
}

bool RemoteStoreOp::hasTensorSemantics() {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return false;
  }

  // Check if the memref operand is a tensor (needs bufferization)
  bool memrefIsTensor = mlir::isa<RankedTensorType>(getMemref().getType());
  // Check if the localBuffer is a tensor (needs bufferization)
  Value localBuffer = getLocalBuffer();
  bool localBufferIsTensor =
      localBuffer && mlir::isa<RankedTensorType>(localBuffer.getType());
  // Check if the result is a tensor (needs bufferization)
  Value result = getResult();
  bool resultIsTensor = result && mlir::isa<RankedTensorType>(result.getType());
  return memrefIsTensor || localBufferIsTensor || resultIsTensor;
}

//===----------------------------------------------------------------------===//
// Index Operations
//===----------------------------------------------------------------------===//

void IterIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "iter" + std::to_string(dim));
}

void IterIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult IterIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

void CoreIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "core" + std::to_string(dim));
}

void CoreIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult CoreIndexOp::fold(FoldAdaptor adaptor) {
  // Only fold to the constant `dim` when no virtualization map is present.
  // If a map is present, the result depends on runtime core coordinates and
  // must not be folded.
  if (adaptor.getPhysToVirtMapAttr()) {
    return {};
  }
  return adaptor.getDimAttr();
}

// TileMatmulBlockOp verification
::mlir::LogicalResult TileMatmulBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getA().getType())) ||
      !llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getB().getType()))) {
    return emitOpError("operands to TileMatmulBlock must have ttcore.tile "
                       "element type");
  }

  return success();
}

void TileMatmulBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getAMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getBMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult TileTilizeBlockOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(getOperation());

  mlir::Value in = getInput();
  mlir::Value out = getOutput();
  if (mlir::isa<mlir::RankedTensorType>(in.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  mlir::Operation *old = getOperation();
  auto newOp = rewriter.create<mlir::tt::d2m::TileTilizeBlockOp>(
      old->getLoc(), out.getType(), in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool TileTilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool TileTilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList TileTilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
TileTilizeBlockOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult TileTilizeBlockOp::verify() {
  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError(
        "operand to TileTilizeBlock must not have ttcore.tile element type");
  }

  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError(
        "output of TileTilizeBlock must have ttcore.tile element type");
  }

  // Verify result type matches output type (DPS style)
  if (getResult().getType() != getOutput().getType()) {
    return emitOpError("result type must match output parameter type");
  }

  return success();
}

void TileTilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult TileUntilizeBlockOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(getOperation());

  mlir::Value in = getInput();
  mlir::Value out = getOutput();
  if (mlir::isa<mlir::RankedTensorType>(in.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  mlir::Operation *old = getOperation();
  auto newOp = rewriter.create<mlir::tt::d2m::TileUntilizeBlockOp>(
      old->getLoc(), out.getType(), in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool TileUntilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool TileUntilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList TileUntilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
TileUntilizeBlockOp::getBufferType(
    mlir::Value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult TileUntilizeBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError(
        "operand to TileUntilizeBlock must have ttcore.tile element type");
  }
  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError(
        "output of TileUntilizeBlock must not have ttcore.tile element type");
  }

  // Verify result type matches output type (DPS style)
  if (getResult().getType() != getOutput().getType()) {
    return emitOpError("result type must match output parameter type");
  }

  return success();
}

void TileUntilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// BlockMaskOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult
BlockMaskOp::bufferize(mlir::RewriterBase &rewriter,
                       const mlir::bufferization::BufferizationOptions &options,
                       mlir::bufferization::BufferizationState &state) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(getOperation());

  mlir::Value in = getInput();
  mlir::Value out = getOutput();
  if (mlir::isa<mlir::RankedTensorType>(in.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  mlir::Operation *old = getOperation();
  auto newOp = rewriter.create<mlir::tt::d2m::BlockMaskOp>(
      old->getLoc(), out.getType(), in, out, getLogicalRows(), getLogicalCols(),
      getFillValue());
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

bool BlockMaskOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool BlockMaskOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList
BlockMaskOp::getAliasingValues(mlir::OpOperand &,
                               const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
BlockMaskOp::getBufferType(mlir::Value,
                           const mlir::bufferization::BufferizationOptions &,
                           const mlir::bufferization::BufferizationState &,
                           ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult BlockMaskOp::verify() {
  // Verify input and output have compatible types.
  auto inType = getInput().getType();
  auto outType = getOutput().getType();

  auto inShapedType = mlir::dyn_cast<mlir::ShapedType>(inType);
  auto outShapedType = mlir::dyn_cast<mlir::ShapedType>(outType);

  if (!inShapedType || !outShapedType) {
    return emitOpError("input and output must be shaped types");
  }

  if (inShapedType.getShape() != outShapedType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  if (inShapedType.getElementType() != outShapedType.getElementType()) {
    return emitOpError("input and output must have the same element type");
  }

  return success();
}

void BlockMaskOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// YieldOp / WaitOp / ReserveOp / PushOp / PopOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  auto generic = getOperation()->getParentOfType<GenericOp>();
  if (!generic || !generic.hasPureTensorSemantics()) {
    return emitOpError()
           << "used outside of generic op with pure tensor semantics";
  }

  return ::mlir::success();
}

bool PushOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool PushOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::bufferization::AliasingValueList
PushOp::getAliasingValues(mlir::OpOperand &,
                          const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
PushOp::getBufferType(mlir::Value,
                      const mlir::bufferization::BufferizationOptions &,
                      const mlir::bufferization::BufferizationState &,
                      ::llvm::SmallVector<mlir::Value> &) {
  llvm_unreachable(
      "intentionally unimplemented, this op can only accept block arguments "
      "which should have already been converted");
}

mlir::LogicalResult
PushOp::bufferize(mlir::RewriterBase &rewriter,
                  const mlir::bufferization::BufferizationOptions &options,
                  mlir::bufferization::BufferizationState &) {
  return bufferizeCBOp(*this, rewriter, options);
}

bool PopOp::bufferizesToMemoryRead(mlir::OpOperand &,
                                   const mlir::bufferization::AnalysisState &) {
  return false;
}

bool PopOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::bufferization::AliasingValueList
PopOp::getAliasingValues(mlir::OpOperand &,
                         const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
PopOp::getBufferType(mlir::Value,
                     const mlir::bufferization::BufferizationOptions &,
                     const mlir::bufferization::BufferizationState &,
                     ::llvm::SmallVector<mlir::Value> &) {
  llvm_unreachable(
      "intentionally unimplemented, this op can only accept block arguments "
      "which should have already been converted");
}

mlir::LogicalResult
PopOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options,
                 mlir::bufferization::BufferizationState &) {
  return bufferizeCBOp(*this, rewriter, options);
}
