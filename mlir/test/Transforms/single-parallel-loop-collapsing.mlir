// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize))' | FileCheck %s

func.func @collapse_to_single() {
  %c0 = arith.constant 3 : index
  %c1 = arith.constant 7 : index
  %c2 = arith.constant 11 : index
  %c3 = arith.constant 29 : index
  %c4 = arith.constant 3 : index
  %c5 = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%c0, %c1) to (%c2, %c3) step (%c4, %c5) {
    %result = "magic.op"(%i0, %i1): (index, index) -> index
  }
  return
}

// CHECK-LABEL: func @collapse_to_single() {
// CHECK-DAG:         [[C18:%.*]] = arith.constant 18 : index
// CHECK-DAG:         [[C6:%.*]] = arith.constant 6 : index
// CHECK-DAG:         [[C3:%.*]] = arith.constant 3 : index
// CHECK-DAG:         [[C7:%.*]] = arith.constant 7 : index
// CHECK-DAG:         [[C4:%.*]] = arith.constant 4 : index
// CHECK-DAG:         [[C1:%.*]] = arith.constant 1 : index
// CHECK-DAG:         [[C0:%.*]] = arith.constant 0 : index
// CHECK:         scf.parallel ([[NEW_I:%.*]]) = ([[C0]]) to ([[C18]]) step ([[C1]]) {
// CHECK:           [[I0_COUNT:%.*]] = arith.remsi [[NEW_I]], [[C6]] : index
// CHECK:           [[I1_COUNT:%.*]] = arith.divsi [[NEW_I]], [[C6]] : index
// CHECK:           [[V0:%.*]] = arith.muli [[I0_COUNT]], [[C4]] : index
// CHECK:           [[I1:%.*]] = arith.addi [[V0]], [[C7]] : index
// CHECK:           [[V1:%.*]] = arith.muli [[I1_COUNT]], [[C3]] : index
// CHECK:           [[I0:%.*]] = arith.addi [[V1]], [[C3]] : index
// CHECK:           "magic.op"([[I0]], [[I1]]) : (index, index) -> index
// CHECK:           scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// -----

func.func @collapse_to_single_reduce(%arg0: memref<?x?xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 0.000000e+00 : f32
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %0 = scf.parallel (%i, %j) = (%c0, %c1) to (%d0, %d1) step (%c1, %c1) init (%c2) -> f32 {
    %0 = memref.load %arg0[%i, %j] : memref<?x?xf32>
    scf.reduce(%0)  : f32 {
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      scf.reduce.return %1 : f32
    }
    scf.yield
  }
  return %0 : f32
}

// CHECK-LABEL: func @collapse_to_single_reduce(
// CHECK-NEXT:    [[C0:%.*]] = arith.constant 0 : index
// CHECK-NEXT:    [[C1:%.*]] = arith.constant 1 : index
// CHECK-NEXT:    [[C2:%.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    [[D0:%.*]] = memref.dim %arg0, [[C0]] : memref<?x?xf32>
// CHECK-NEXT:    [[D1:%.*]] = memref.dim %arg0, [[C1]] : memref<?x?xf32>
// CHECK-NEXT:    [[X:%.*]] = arith.subi [[D1]], [[C1]] : index
// CHECK-NEXT:    [[Y:%.*]] = arith.muli [[D0]], [[X]] : index
// CHECK-NEXT:    [[V0:%.*]] = scf.parallel ([[NEW_I:%.*]]) = ([[C0]]) to ([[Y]]) step ([[C1]]) init ([[C2]]) -> f32 {
// CHECK-NEXT:      [[I:%.*]] = arith.remsi [[NEW_I]], [[X]] : index
// CHECK-NEXT:      [[J:%.*]] = arith.divsi [[NEW_I]], [[X]] : index
// CHECK-NEXT:      [[K:%.*]] = arith.addi [[I]], [[C1]] : index
// CHECK-NEXT:      [[V1:%.*]] = memref.load %arg0[[[J]], [[K]]] : memref<?x?xf32>
// CHECK-NEXT:      scf.reduce([[V1]])  : f32 {
// CHECK-NEXT:      ^bb0([[V2:%.*]]: f32, [[V3:%.*]]: f32):
// CHECK-NEXT:        [[V4:%.*]] = arith.addf [[V2]], [[V3]] : f32
// CHECK-NEXT:        scf.reduce.return [[V4]] : f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return [[V0]] : f32
// CHECK-NEXT:  }
