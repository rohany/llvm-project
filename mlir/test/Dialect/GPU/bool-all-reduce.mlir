// RUN: mlir-opt --gpu-kernel-outlining --convert-gpu-to-nvvm %s | FileCheck %s

gpu.module @mod {
  gpu.func @kernel(%arg0: memref<i1>) kernel {
    %0 = memref.load %arg0[] : memref<i1>
    %1 = gpu.all_reduce  or %0 uniform {
    } : (i1) -> i1
    gpu.return
  }
}

// CHECK:      gpu.module @mod {
// CHECK-NEXT:   llvm.mlir.global internal @{{.*}}() {addr_space = 3 : i32} : !llvm.array<32 x i1>
// CHECK-NEXT:   llvm.func @{{.*}}(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) attributes {gpu.kernel, nvvm.kernel} {
// CHECK:          ^bb1:
// CHECK:            [[EXT:%.+]] = llvm.zext %{{.*}} : i1 to i32
// CHECK:            [[SHFL:%.+]] = nvvm.shfl.sync  bfly %{{.*}}, [[EXT]]
// CHECK-NEXT:       [[SHFL_RES:%.*]] = llvm.extractvalue [[SHFL]][0]
// CHECK:            llvm.trunc [[SHFL_RES]] : i32 to i1
