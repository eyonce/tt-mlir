// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-generate-datamovement -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#reduction = #ttcore.iterator_type<reduction>


func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
  %0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #ttcore.view<map(4)>, #l1_>
  %1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: d2m.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "d2m.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // Look for 4 regions, one for each operand and one for the compute
  // Operand 0 (input)
  // CHECK: ^datamovement0
  // CHECK: d2m.dma [[lhs]]<#map1>
  // CHECK-NEXT: d2m.dma_wait
  // CHECK-NEXT: d2m.semaphore_wait [[reader_ready_lhs:%[a-z0-9]+]]
  // CHECK: d2m.dma {{%.*}}, {{%.*}}
  // CHECK-NEXT: d2m.dma_wait
  // CHECK-NEXT: d2m.semaphore_set [[writer_done_lhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: d2m.semaphore_inc [[reader_ready_lhs]]
  // CHECK-NEXT: d2m.semaphore_wait [[writer_done_lhs]]
  // Operand 1 (input)
  // CHECK: ^datamovement1
  // CHECK: d2m.dma [[rhs]]<#map2>
  // CHECK-NEXT: d2m.dma_wait
  // CHECK-NEXT: d2m.semaphore_wait [[reader_ready_rhs:%[a-z0-9]+]]
  // CHECK: d2m.dma {{%.*}}, {{%.*}}
  // CHECK-NEXT: d2m.dma_wait
  // CHECK-NEXT: d2m.semaphore_set [[writer_done_rhs:%[a-z0-9]+]]
  // CHECK-NEXT: else
  // CHECK-NEXT: d2m.semaphore_inc [[reader_ready_rhs]]
  // CHECK-NEXT: d2m.semaphore_wait [[writer_done_rhs]]
  // Operand 2 (output)
  // CHECK: ^datamovement2
  // Compute
  // CHECK: ^compute
  // CHECK: d2m.tile_matmul_block
  ^bb0(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
}
