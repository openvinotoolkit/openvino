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
//     nvidia/compat_wg.cl       work-group collectives
// Each file relies only on the ones before it.

#ifdef __NV_CL_C_VERSION

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef cl_khr_fp16
#define cl_khr_fp16 1
#endif
// The host reports supports_intel_subgroups_short for NVIDIA devices (sycl_device.cpp), so the kernel selector
// can pick kernels whose whole body sits under #if defined(cl_intel_subgroups_short) (convolution_gpu_bfyx_direct_
// 10_12_16, convolution_gpu_bfyx_gemm_like_fp16). The NVIDIA compiler never defines the macro, so without this those
// kernels preprocess to nothing and the batch comes back one entry point short. The 16-bit block accessors the
// extension would supply are the intel_sub_group_block_{read,write}_us* macros defined in compat_sg.cl, so
// sub_group_block_read.cl and sub_group_block_write.cl reach the same helpers either way; compat_sg_overrides.cl
// then #undefs the _sub_group_block_*_us* spellings those headers define and binds them again.
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
// The helpers below replace these built-ins, and the other work-item ones, and read the special registers instead -
// directly, or through placeholders the runtime replaces with them (see "Work-group layout on CUDA" below); OpenVINO
// always launches with a zero global offset. Trying the built-ins through clEnqueueNDRangeKernel
// does not show the problem, since the OpenCL runtime does set %envreg9 there.
// The asm is not volatile, here and for _nv_laneid in compat_sg.cl: these registers do not change while a work-item
// runs, so the compiler may fold repeated reads, which cuts the S2R count. The code shape this gives exposed three
// problems on an A100 with the sub-group size emulation on, each fixed where it lies rather than by keeping volatile:
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

// Work-group layout on CUDA.
// CUDA allows at most 64 threads along blockDim.z, while an OpenCL work-group may have up to max_work_group_size
// work-items along any dimension. sycl_ptx_kernel.cpp therefore launches a work-group of local size (X, Y, Z) with
// Z > 64 as the CUDA block (X, Z, Y): it folds dimensions 1 and 2 into one axis of Y * Z work-items, counted by
// r = tid.y + ntid.y * tid.z, and keeps Y in %ntid.z (Y * Z <= 1024 and Z > 64 leave Y <= 15). The hardware still
// numbers the threads x fastest, so r and the linear local id - and with them the warp lanes the sub-group emulation
// relies on - are those of the OpenCL work-group, and the local ids are recovered as r % Y and r / Y. When Y is 1
// that is a plain exchange of the two axes and needs no division.
//
// So that the same PTX serves every layout, the helpers below do not read the values that depend on it from
// special registers but from %envreg27-31, which cuLaunchKernel never sets and the NVIDIA OpenCL built-ins do not use
// (they use %envreg0-9). When sycl_ptx_kernel.cpp loads the PTX it replaces each of them with the special register
// the chosen layout calls for, so the asm here never reaches the hardware as written:
//     %envreg27-30  local id 1, local id 2, local size 1, local size 2: the axes they sit on in the layout.
//     %envreg31     the divisor of the fold: %ntid.z when Y > 1 is folded, %ntid.y otherwise. Compared with %ntid.y
//                   it selects the division, and in the other layouts ptxas folds the comparison and drops it.
// The runtime folds a program only when its PTX has %envreg31, which shows it was built with these helpers: the
// OpenCL built-ins they replace would read the wrong axis once the layout moves dimensions 1 and 2. %envreg27-30
// only spare the division when Y is 1 and could be dropped first, so they take the lower numbers. Dimension 0, the
// hardware y and z values and the work-group ids and counts are read from their special registers directly: the
// layout never moves dimension 0, the hardware numbering is what the fold is built on, and the grid axes are
// remapped by renaming %ctaid/%nctaid.
static inline uint _nv_hw_tid_y(void)      { uint v; __asm__("mov.u32 %0, %%tid.y;"  : "=r"(v)); return v; }
static inline uint _nv_hw_tid_z(void)      { uint v; __asm__("mov.u32 %0, %%tid.z;"  : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_y(void)     { uint v; __asm__("mov.u32 %0, %%ntid.y;" : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_z(void)     { uint v; __asm__("mov.u32 %0, %%ntid.z;" : "=r"(v)); return v; }
static inline uint _nv_local_id_1(void)    { uint v; __asm__("mov.b32 %0, %%envreg27;" : "=r"(v)); return v; }
static inline uint _nv_local_id_2(void)    { uint v; __asm__("mov.b32 %0, %%envreg28;" : "=r"(v)); return v; }
static inline uint _nv_local_size_1(void)  { uint v; __asm__("mov.b32 %0, %%envreg29;" : "=r"(v)); return v; }
static inline uint _nv_local_size_2(void)  { uint v; __asm__("mov.b32 %0, %%envreg30;" : "=r"(v)); return v; }
static inline uint _nv_relin_divisor(void) { uint v; __asm__("mov.b32 %0, %%envreg31;" : "=r"(v)); return v; }

static inline uint _nv_hw_tid_x(void)  { uint v; __asm__("mov.u32 %0, %%tid.x;"  : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_x(void) { uint v; __asm__("mov.u32 %0, %%ntid.x;" : "=r"(v)); return v; }

static inline uint _nv_group_id(uint dim) {
    uint v = 0;
    if (dim == 0) __asm__("mov.u32 %0, %%ctaid.x;" : "=r"(v));
    else if (dim == 1) __asm__("mov.u32 %0, %%ctaid.y;" : "=r"(v));
    else if (dim == 2) __asm__("mov.u32 %0, %%ctaid.z;" : "=r"(v));
    return v;
}

static inline uint _nv_num_groups(uint dim) {
    uint v = 1;
    if (dim == 0) __asm__("mov.u32 %0, %%nctaid.x;" : "=r"(v));
    else if (dim == 1) __asm__("mov.u32 %0, %%nctaid.y;" : "=r"(v));
    else if (dim == 2) __asm__("mov.u32 %0, %%nctaid.z;" : "=r"(v));
    return v;
}

// OpenCL's get_local_id().
static inline uint _nv_local_id(uint dim) {
    if (dim == 0) return _nv_hw_tid_x();
    if (dim > 2) return 0;
    // Y > 1 folded with Z: the ids from the position along the folded axis. They are computed outside any branch and
    // picked with selects, so that all the calls in a function share one division (a division in a branch per call
    // site stays one per call site). In the other layouts the selects fold and ptxas drops the division.
    const uint d = _nv_relin_divisor();
    const uint r = _nv_hw_tid_y() + _nv_hw_ntid_y() * _nv_hw_tid_z();
    const uint q = r / d;
    const uint m = r - q * d;
    const bool folded = d != _nv_hw_ntid_y();
    return dim == 1 ? (folded ? m : _nv_local_id_1()) : (folded ? q : _nv_local_id_2());
}

// OpenCL's get_local_size().
static inline uint _nv_local_size(uint dim) {
    if (dim == 0) return _nv_hw_ntid_x();
    if (dim == 1) return _nv_local_size_1();
    if (dim == 2) return _nv_local_size_2();
    return 1;
}

// OpenCL's get_global_id() and get_global_size() with a zero global offset, formed in size_t so that NDRanges of
// more than 2^32 work-items do not wrap.
static inline size_t _nv_global_id(uint dim) {
    if (dim > 2) return 0;
    return (size_t)_nv_group_id(dim) * _nv_local_size(dim) + _nv_local_id(dim);
}

static inline size_t _nv_global_size(uint dim) {
    if (dim > 2) return 1;
    return (size_t)_nv_num_groups(dim) * _nv_local_size(dim);
}

// Linear id of the work-item inside its work-group, dim 0 fastest (OpenCL's get_local_linear_id()). The hardware
// numbering is the OpenCL one in every layout (see above), so it is formed from the hardware registers.
static inline uint _nv_get_linear_tid(void){
    return _nv_hw_tid_x() + _nv_hw_ntid_x() * (_nv_hw_tid_y() + _nv_hw_ntid_y() * _nv_hw_tid_z());
}

// Number of work-items in the work-group.
static inline uint _nv_get_total_threads(void){
    return _nv_hw_ntid_x() * _nv_hw_ntid_y() * _nv_hw_ntid_z();
}

// Linear id of the work-item in the NDRange, dim 0 fastest (OpenCL's get_global_linear_id() with a zero offset).
static inline size_t _nv_get_global_linear_id(void){
    return _nv_global_id(0) + _nv_global_size(0) * (_nv_global_id(1) + _nv_global_size(1) * _nv_global_id(2));
}

#define get_local_id(dim)            ((size_t)_nv_local_id(dim))
#define get_local_size(dim)          ((size_t)_nv_local_size(dim))
#define get_enqueued_local_size(dim) ((size_t)_nv_local_size(dim))
#define get_global_id(dim)           _nv_global_id(dim)
#define get_global_size(dim)         _nv_global_size(dim)
#define get_local_linear_id()        ((size_t)_nv_get_linear_tid())
#define get_global_linear_id()       _nv_get_global_linear_id()

#endif // __NV_CL_C_VERSION
