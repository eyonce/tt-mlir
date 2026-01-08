// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDECOUPLELOADSTOREOPSFROMCOMPUTE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is remote (i.e., comes from a stream
// op, or is used in a DMA-only GenericOp where all operands are considered
// remote)
static bool isRemoteOperand(Value operand, Operation *op) {
  // Check if the operation is inside a DMA-only generic op
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (generic && generic.isDMAOnlyForm()) {
    // In DMA-only form, all operands are considered remote
    return true;
  }

  // Remote operands are those that come from stream_layout ops
  return mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp());
}

// Helper function to find the CB block argument that corresponds to a memref
// operand in a generic op. Returns the CB block argument if found, null
// otherwise. Assumes that the operand index in the generic op equals the CB
// block arg index.
//
// For DMA-only GenericOps with remote loads: returns the CB associated with
// the output operand (destination), not the input operand being loaded.
static Value findAssociatedCB(Operation *op, Value memrefOperand) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find which operand index this memref corresponds to
  unsigned operandIndex = UINT_MAX;
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == memrefOperand) {
      operandIndex = i;
      break;
    }
  }

  if (operandIndex == UINT_MAX) {
    return Value();
  }

  // Special case: For DMA-only GenericOps with remote loads, use the output
  // operand's CB instead of the input operand's CB
  if (generic.isDMAOnlyForm() && mlir::isa<RemoteLoadOp>(op)) {
    // In DMA-only form, we load into the output operand's CB
    // The output operands come after the input operands in the operand list
    unsigned numInputs = generic.getInputs().size();
    if (generic.getOutputs().size() > 0) {
      // Use the first output operand's index
      operandIndex = numInputs;
    }
  }

  // Find the generic op's thread region that contains this operation
  // If there's only one region, use it directly. Otherwise, use the utility
  // function
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }

  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  // Get the first block of the generic region (thread region block)
  Block *threadBlock = &genericRegion->front();

  // The CB block arguments are in the same order as the generic operands
  // The operand index equals the CB block arg index
  if (threadBlock->getNumArguments() > operandIndex) {
    return threadBlock->getArgument(operandIndex);
  }

  return Value();
}

// Helper function to find the CB block argument by operand index directly
static Value findCBByOperandIndex(Operation *op, unsigned operandIndex) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find the generic op's thread region that contains this operation
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }

  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  Block *threadBlock = &genericRegion->front();

  if (threadBlock->getNumArguments() > operandIndex) {
    return threadBlock->getArgument(operandIndex);
  }

  return Value();
}

// Helper function to find the ReserveOp that produces a given value,
// potentially through a chain of operations. This is used after we've
// converted AcquireBufferOp to ReserveOp to find the associated CB.
static ReserveOp findReserveOp(Value value) {
  if (!value) {
    return nullptr;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }

  // Direct case: value is directly produced by reserve
  if (auto reserveOp = mlir::dyn_cast<ReserveOp>(definingOp)) {
    return reserveOp;
  }

  // Trace through operations that might pass the buffer through
  for (Value operand : definingOp->getOperands()) {
    if (auto reserveOp = findReserveOp(operand)) {
      return reserveOp;
    }
  }

  return nullptr;
}

class D2MDecoupleLoadStoreOpsFromCompute
    : public impl::D2MDecoupleLoadStoreOpsFromComputeBase<
          D2MDecoupleLoadStoreOpsFromCompute> {
public:
  using impl::D2MDecoupleLoadStoreOpsFromComputeBase<
      D2MDecoupleLoadStoreOpsFromCompute>::
      D2MDecoupleLoadStoreOpsFromComputeBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // PHASE 1: Transform RemoteLoadOp (implicit form -> explicit CB form)
    // Collect remote_load operations to convert (those with remote operands
    // and implicit form - i.e., they have a result)
    SmallVector<RemoteLoadOp> remoteLoadsToConvert;
    moduleOp->walk([&](RemoteLoadOp remoteLoad) {
      Value memref = remoteLoad.getMemref();
      // Only handle remote operands (from stream_layout ops)
      if (!isRemoteOperand(memref, remoteLoad.getOperation())) {
        return;
      }
      // Only handle implicit form (has result, no CB operand)
      if (!remoteLoad.isImplicitForm()) {
        return;
      }
      remoteLoadsToConvert.push_back(remoteLoad);
    });

    // Transform each remote_load
    for (RemoteLoadOp remoteLoad : remoteLoadsToConvert) {
      Location loc = remoteLoad.getLoc();
      Value memref = remoteLoad.getMemref();
      Value assocCb = findAssociatedCB(remoteLoad.getOperation(), memref);

      if (!assocCb) {
        remoteLoad.emitWarning(
            "could not find associated CB block argument, skipping conversion");
        continue;
      }

      rewriter.setInsertionPoint(remoteLoad);

      // Create the explicit CB form of remote_load (no result, has CB operand)
      // d2m.remote_load %memref[indices] into %cb
      rewriter.create<RemoteLoadOp>(
          loc, assocCb, memref, remoteLoad.getIndices(),
          remoteLoad.getMcastStartIndex(), remoteLoad.getMcastShape());

      // Create wait operation to produce the result value
      // %in = d2m.wait %cb
      auto waitOp = rewriter.create<WaitOp>(loc, assocCb);

      // Replace all uses of remote_load result with wait result
      if (remoteLoad.getResult()) {
        rewriter.replaceAllUsesWith(remoteLoad.getResult(), waitOp.getResult());
      }

      // Insert pop at the end of the top-level region scope (before terminator)
      // to ensure it's outside any nested loops
      GenericOp generic = remoteLoad->getParentOfType<GenericOp>();
      Region *genericRegion = nullptr;
      if (generic.getNumRegions() == 1) {
        genericRegion = &generic.getRegion(0);
      } else {
        genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(
            remoteLoad.getOperation());
      }

      if (genericRegion && !genericRegion->empty()) {
        Block *topLevelBlock = &genericRegion->front();
        // Insert before the terminator (YieldOp)
        if (!topLevelBlock->empty() &&
            topLevelBlock->back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&topLevelBlock->back());
        } else {
          rewriter.setInsertionPointToEnd(topLevelBlock);
        }
        rewriter.create<PopOp>(loc, assocCb);
      } else {
        remoteLoad.emitWarning(
            "could not find top-level region block for pop insertion");
      }

      // Erase the original remote_load operation
      rewriter.eraseOp(remoteLoad);
    }

    // PHASE 2a: Convert AcquireBufferOp -> ReserveOp (but don't insert push
    // yet) We need to do this first so RemoteStoreOp can find the CB via the
    // reserve op. The push will be inserted in PHASE 4 after remote_stores are
    // converted.
    SmallVector<std::pair<ReserveOp, Value>> reserveOpsCreated;

    SmallVector<AcquireBufferOp> acquireBuffersToConvert;
    moduleOp->walk([&](AcquireBufferOp acquireBuffer) {
      // Check if it has an operand_index attribute
      if (!acquireBuffer.getOperandIndex().has_value()) {
        return;
      }

      unsigned operandIndex = acquireBuffer.getOperandIndex().value();
      GenericOp generic = acquireBuffer->getParentOfType<GenericOp>();
      if (!generic) {
        return;
      }

      // Check if the corresponding generic operand is remote
      if (operandIndex >= generic->getNumOperands()) {
        return;
      }

      Value genericOperand = generic->getOperand(operandIndex);
      if (!isRemoteOperand(genericOperand, acquireBuffer.getOperation())) {
        return;
      }

      acquireBuffersToConvert.push_back(acquireBuffer);
    });

    // Transform each acquire_buffer to reserve (don't insert push yet)
    for (AcquireBufferOp acquireBuffer : acquireBuffersToConvert) {
      Location loc = acquireBuffer.getLoc();
      unsigned operandIndex = acquireBuffer.getOperandIndex().value();

      Value assocCb =
          findCBByOperandIndex(acquireBuffer.getOperation(), operandIndex);
      if (!assocCb) {
        acquireBuffer.emitWarning(
            "could not find associated CB block argument, skipping conversion");
        continue;
      }

      rewriter.setInsertionPoint(acquireBuffer);

      // Create reserve operation
      // %out = d2m.reserve %cb
      auto reserveOp = rewriter.create<ReserveOp>(loc, assocCb);

      // Replace all uses of acquire_buffer result with reserve result
      rewriter.replaceAllUsesWith(acquireBuffer.getResult(),
                                  reserveOp.getResult());

      // Remember this reserve op and its CB for later push insertion
      reserveOpsCreated.push_back({reserveOp, assocCb});

      // Erase the original acquire_buffer operation
      rewriter.eraseOp(acquireBuffer);
    }

    // PHASE 3: Transform RemoteStoreOp (implicit form -> explicit CB form)
    // This must happen BEFORE we insert push ops, so that the remote_store
    // no longer counts as a use of the reserve result.
    SmallVector<RemoteStoreOp> remoteStoresToConvert;
    moduleOp->walk([&](RemoteStoreOp remoteStore) {
      Value memref = remoteStore.getMemref();
      // Only handle remote operands (from stream_layout ops)
      if (!isRemoteOperand(memref, remoteStore.getOperation())) {
        return;
      }
      // Only handle implicit form (has local buffer, no CB operand)
      if (!remoteStore.isImplicitForm()) {
        return;
      }
      remoteStoresToConvert.push_back(remoteStore);
    });

    // Transform each remote_store
    for (RemoteStoreOp remoteStore : remoteStoresToConvert) {
      Location loc = remoteStore.getLoc();
      Value memref = remoteStore.getMemref();
      Value localBuffer = remoteStore.getLocalBuffer();

      // Find the CB associated with the local buffer by tracing back to the
      // reserve op that produced it
      ReserveOp reserveOp = findReserveOp(localBuffer);
      if (!reserveOp) {
        remoteStore.emitWarning(
            "could not find reserve op for local buffer, skipping conversion");
        continue;
      }

      Value assocCb = reserveOp.getCb();

      rewriter.setInsertionPoint(remoteStore);

      // Create the explicit CB form of remote_store (no local buffer, has CB)
      // d2m.remote_store %memref[indices] from %cb
      rewriter.create<RemoteStoreOp>(loc, memref, remoteStore.getIndices(),
                                     assocCb, remoteStore.getMcastStartIndex(),
                                     remoteStore.getMcastShape());

      // Erase the original remote_store operation
      rewriter.eraseOp(remoteStore);
    }

    // PHASE 4: Insert push ops for each reserve op
    // Insert push at the end of the top-level region scope (before terminator)
    // to ensure it's outside any nested loops
    for (auto &[reserveOp, assocCb] : reserveOpsCreated) {
      Location loc = reserveOp.getLoc();

      GenericOp generic = reserveOp->getParentOfType<GenericOp>();
      Region *genericRegion = nullptr;
      if (generic.getNumRegions() == 1) {
        genericRegion = &generic.getRegion(0);
      } else {
        genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(
            reserveOp.getOperation());
      }

      if (genericRegion && !genericRegion->empty()) {
        Block *topLevelBlock = &genericRegion->front();
        // Insert before the terminator (YieldOp)
        if (!topLevelBlock->empty() &&
            topLevelBlock->back().hasTrait<OpTrait::IsTerminator>()) {
          rewriter.setInsertionPoint(&topLevelBlock->back());
        } else {
          rewriter.setInsertionPointToEnd(topLevelBlock);
        }
        rewriter.create<PushOp>(loc, assocCb);
      } else {
        reserveOp.emitWarning(
            "could not find top-level region block for push insertion");
      }
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
