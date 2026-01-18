// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

!ttype = tensor<128x96xf32>
!ttype_col = tensor<128x1xf32>
!ttype_row = tensor<1x96xf32>
!ttype_scalar = tensor<1x1xf32>

!lhs = tensor<128x96xf32>
!rhs = tensor<96x64xf32>
!matmul_result = tensor<128x64xf32>

module {

  func.func @binary_eltwise(%lhs: !ttype, %rhs: !ttype, %out: !ttype) -> (!ttype) {
    %0 = "ttir.add"(%lhs, %rhs) : (!ttype, !ttype) -> !ttype
    return %0: !ttype
  }

  func.func @unary_eltwise(%lhs: !ttype, %rhs: !ttype, %out: !ttype) -> (!ttype) {
    %1 = "ttir.exp"(%lhs) : (!ttype) -> !ttype
    return %1: !ttype
  }

  func.func @named_ternary_where(%cond: !ttype, %true_val: !ttype, %false_val: !ttype) -> (!ttype) {
    %0 = "ttir.where"(%cond, %true_val, %false_val) : (!ttype, !ttype, !ttype) -> !ttype
    return %0 : !ttype
  }

  func.func @named_reductions_R(%arg: !ttype) -> (tensor<1x96xf32>) {
    %1 = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (!ttype) -> tensor<1x96xf32>
    return %1: tensor<1x96xf32>
  }

  func.func @named_reductions_C(%arg: !ttype) -> (tensor<128x1xf32>) {
    %1 = "ttir.sum"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (!ttype) -> tensor<128x1xf32>
    return %1 : tensor<128x1xf32>
  }

  func.func @named_reductions_RC(%arg: !ttype) -> (tensor<1x1xf32>) {
    %1 = "ttir.sum"(%arg) <{dim_arg = [-2: i32, -1: i32], keep_dim = true}> : (!ttype) -> tensor<1x1xf32>
    return %1: tensor<1x1xf32>
  }

  func.func @named_contractions(%lhs: !lhs, %rhs: !rhs) -> (!matmul_result) {
    %r = "ttir.matmul"(%lhs, %rhs) : (!lhs, !rhs) -> (!matmul_result)
    return %r : !matmul_result
  }

  func.func @implicit_bcast_2d_dual(%in0: !ttype_col, %in1: !ttype_row) -> (!ttype) {
    %0 = "ttir.add"(%in0, %in1) : (!ttype_col, !ttype_row) -> (!ttype)
    return %0 : !ttype
  }

  func.func @implicit_bcast_2d_scalar(%in0: !ttype, %in1: !ttype_scalar) -> (!ttype) {
    %0 = "ttir.add"(%in0, %in1) : (!ttype, !ttype_scalar) -> (!ttype)
    return %0 : !ttype
  }

  func.func @named_slice_static(%arg0: tensor<96x96xf32>) -> tensor<32x32xf32> {
    %0 = "ttir.slice_static"(%arg0) <{begins = [1 : i32, 0 : i32], ends = [96 : i32, 64 : i32], step = [3 : i32, 2 : i32]}> : (tensor<96x96xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
