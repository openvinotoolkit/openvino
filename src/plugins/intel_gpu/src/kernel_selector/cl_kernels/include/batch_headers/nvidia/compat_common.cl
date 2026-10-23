// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Common base of the NVIDIA compatibility prelude: extension pragmas, feature macros and work-item id helpers.
// Keep it to the extension / feature-macro setup and to helpers shared by several of the later compat_*.cl files;
// code used by a single feature belongs in that feature's file.
//
// kernels_cache.cpp emits the prelude as the concatenation of nvidia_prelude_names, in this order:
//     nvidia/compat_common.cl   this file
//     nvidia/compat_half.cl     half limits, math, conversions, reinterprets, vload/vstore, relational
//     nvidia/compat_sg.cl       sub-group emulation
// Each file relies only on the ones before it.

#ifdef __NV_CL_C_VERSION

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef cl_khr_fp16
#define cl_khr_fp16 1
#endif
// The host reports supports_intel_subgroups_short when OV_NV_EMULATE_SUBGROUP_SIZES is on, so the kernel selector
// can pick kernels whose whole body sits under #if defined(cl_intel_subgroups_short) (convolution_gpu_bfyx_direct_
// 10_12_16, convolution_gpu_bfyx_gemm_like_fp16). The NVIDIA compiler never defines the macro, so without this those
// kernels preprocess to nothing and the batch comes back one entry point short. The 16-bit block accessors the
// extension would supply are the intel_sub_group_block_{read,write}_us* macros defined in compat_sg.cl, so
// sub_group_block_read.cl and sub_group_block_write.cl reach the same helpers either way; compat_sg_overrides.cl
// then #undefs the _sub_group_block_*_us* spellings those headers define and binds them again. With the emulation off
// the host flag stays false and none of these kernels is selected.
#ifndef cl_intel_subgroups_short
#define cl_intel_subgroups_short 1
#endif

// Work-item ids.
// OpenVINO builds the kernels with clBuildProgram but launches the resulting PTX with cuLaunchKernel, which never
// writes the %envreg special registers the NVIDIA OpenCL runtime sets up. sycl_kernel_builder.cpp patches %envreg0-8
// (global offsets and group counts) in the PTX, but not %envreg9, the work dimension. The built-in
// get_global_linear_id() and get_local_linear_id() branch on that register (one, two or three dimensions) and fall
// through to 0 when it holds anything else, so every work-item got linear id 0: grid_sample_opt_bilinear_zeros, with a
// {1, 256, 1} local size, had all its work-items handle the first element and left the rest of the output at 0.
// The helpers below read the hardware special registers instead, which are set however the kernel is launched and need
// no patching; OpenVINO always launches with a zero global offset. Trying the built-ins through clEnqueueNDRangeKernel
// does not show the problem, since the OpenCL runtime does set %envreg9 there.
// The asm is not volatile, here and for _nv_laneid in compat_sg.cl: these registers do not change while a work-item
// runs, so the compiler may fold repeated reads, which cuts the S2R count. The code shape this gives exposed three
// problems on an A100 with OV_NV_EMULATE_SUBGROUP_SIZES=1, each fixed where it lies rather than by keeping volatile:
//  - convolution_gpu_mmad_bfyx_to_b_fs_yx_fsv32: with the lane and sub-group ids from one %tid read, the x-boundary
//    test was unswitched out of the SLM fill loop and a race between the SLM stores of two sub-groups returned wrong
//    values (fusings_gpu/conv_int8_*). The kernel now guards those stores.
//  - convolution_gpu_bfyx_f16 / _1x1 with fused eltwise ops: ptxas contracted a half multiply and add into HFMA2,
//    1 ulp away from the unfused reference (conv_fp32_eltwise_fusing_extend_ops.pattern03_sub_div/3 and /4). The
//    test's f16 tolerance now allows for that.
//  - sdpa_opt__multi_tokens: its register count rose past the 128 a 512-work-item work-group can use, and the launch
//    failed (smoke_adaptive_rkv/adaptive_rkv_diversity_test.basic/23). sycl_ptx_kernel now loads a kernel that does
//    not fit its work-group again with CU_JIT_THREADS_PER_BLOCK.
// A kernel whose SLM accesses between work-items are not synchronised can still change behaviour with the code shape;
// compute-sanitizer --tool racecheck reports such accesses even when the results happen to be right.

// Linear id of the work-item inside its work-group, dim 0 fastest (OpenCL's get_local_linear_id()).
static inline uint _nv_get_linear_tid(void){
    uint bdx, bdy;
    uint tdx, tdy, tdz;
    __asm__("mov.u32 %0, %%ntid.x;" : "=r"(bdx));
    __asm__("mov.u32 %0, %%ntid.y;" : "=r"(bdy));
    __asm__("mov.u32 %0, %%tid.x;"  : "=r"(tdx));
    __asm__("mov.u32 %0, %%tid.y;"  : "=r"(tdy));
    __asm__("mov.u32 %0, %%tid.z;"  : "=r"(tdz));
    uint linear = (bdx * bdy) * tdz + bdx * tdy + tdx;
    return linear;
}

// Number of work-items in the work-group.
static inline uint _nv_get_total_threads(void){
    uint nx, ny, nz;
    __asm__("mov.u32 %0, %%ntid.x;" : "=r"(nx));
    __asm__("mov.u32 %0, %%ntid.y;" : "=r"(ny));
    __asm__("mov.u32 %0, %%ntid.z;" : "=r"(nz));
    return nx * ny * nz;
}

// Linear id of the work-item in the NDRange, dim 0 fastest (OpenCL's get_global_linear_id() with a zero offset). The
// products are formed in size_t so that NDRanges of more than 2^32 work-items do not wrap.
static inline size_t _nv_get_global_linear_id(void){
    uint tx, ty, tz, nx, ny, nz, cx, cy, cz, gx, gy;
    __asm__("mov.u32 %0, %%tid.x;"    : "=r"(tx));
    __asm__("mov.u32 %0, %%tid.y;"    : "=r"(ty));
    __asm__("mov.u32 %0, %%tid.z;"    : "=r"(tz));
    __asm__("mov.u32 %0, %%ntid.x;"   : "=r"(nx));
    __asm__("mov.u32 %0, %%ntid.y;"   : "=r"(ny));
    __asm__("mov.u32 %0, %%ntid.z;"   : "=r"(nz));
    __asm__("mov.u32 %0, %%ctaid.x;"  : "=r"(cx));
    __asm__("mov.u32 %0, %%ctaid.y;"  : "=r"(cy));
    __asm__("mov.u32 %0, %%ctaid.z;"  : "=r"(cz));
    __asm__("mov.u32 %0, %%nctaid.x;" : "=r"(gx));
    __asm__("mov.u32 %0, %%nctaid.y;" : "=r"(gy));
    const size_t idx = (size_t)cx * nx + tx;
    const size_t idy = (size_t)cy * ny + ty;
    const size_t idz = (size_t)cz * nz + tz;
    return idx + (size_t)gx * nx * (idy + (size_t)gy * ny * idz);
}

#define get_local_linear_id()  ((size_t)_nv_get_linear_tid())
#define get_global_linear_id() _nv_get_global_linear_id()

#endif // __NV_CL_C_VERSION
