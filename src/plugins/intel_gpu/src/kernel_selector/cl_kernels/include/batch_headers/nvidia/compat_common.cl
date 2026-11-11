// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Common base of the NVIDIA compatibility prelude: extension pragmas, feature macros and the work-item id helpers the
// later compat_*.cl files share; code used by a single feature belongs in that feature's file.
//
// kernels_cache.cpp puts the prelude ahead of the batch headers, in the order of nvidia_prelude_names:
//     nvidia/compat_common.cl   this file
//     nvidia/compat_half.cl     half limits, math, conversions, reinterprets, vload/vstore, relational, atomics
//     nvidia/compat_sg.cl       sub-group emulation
//     nvidia/compat_wg.cl       work-group collectives
// Each file relies only on the ones before it, and lists them instead of #including them: primitive_db_gen.py strips
// #include from batch headers, and its dependency sort fails on include graphs that reach a header along two paths.

#ifdef __NV_CL_C_VERSION

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef cl_khr_fp16
#define cl_khr_fp16 1
#endif
// sycl_device.cpp reports supports_intel_subgroups_short for NVIDIA devices, so kernels whose body sits under
// #if defined(cl_intel_subgroups_short) can be picked; the NVIDIA compiler never defines the macro, and without it those
// kernels would preprocess to nothing. compat_sg.cl provides the 16-bit accessors the extension would.
#ifndef cl_intel_subgroups_short
#define cl_intel_subgroups_short 1
#endif

// Work-item ids.
// The PTX is launched with cuLaunchKernel, which never sets the %envreg registers the NVIDIA OpenCL runtime fills in.
// sycl_kernel_builder.cpp patches %envreg0-8, but not %envreg9 (the work dimension) on which the built-in
// get_global_linear_id() / get_local_linear_id() branch, so the helpers below replace the work-item built-ins and read
// the special registers instead, directly or through the layout placeholders below. OpenVINO always launches with a
// zero global offset.
// The asm is not volatile, here and for _nv_laneid in compat_sg.cl: these registers do not change while a work-item
// runs, so repeated reads may be folded. Kernels whose SLM accesses are not synchronised can change behaviour with it.

// Work-group layout on CUDA.
// CUDA allows at most 64 threads along blockDim.z, so sycl_ptx_kernel.cpp launches a work-group of local size
// (X, Y, Z) with Z > 64 as the CUDA block (X, Z, Y): dimensions 1 and 2 fold into one axis of Y * Z work-items,
// counted by r = tid.y + ntid.y * tid.z, with Y kept in %ntid.z. The hardware still numbers the threads x fastest, so
// the linear local id - and the warp lanes the sub-group emulation relies on - stay those of the OpenCL work-group,
// and the local ids are r % Y and r / Y (a plain exchange of the two axes when Y is 1).
//
// So that one PTX serves every layout, the values that depend on it are read from %envreg27-31, which neither
// cuLaunchKernel nor the NVIDIA built-ins use; sycl_ptx_kernel.cpp replaces each with the special register the chosen
// layout calls for when it loads the PTX:
//     %envreg27-30  local id 1, local id 2, local size 1, local size 2.
//     %envreg31     the divisor of the fold: %ntid.z when Y > 1 is folded, %ntid.y otherwise. Compared with %ntid.y
//                   it selects the division, which ptxas drops in the other layouts.
// Dimension 0, the hardware y and z values and the work-group ids and counts are read from their special registers
// directly: the layout never moves dimension 0, the fold is built on the hardware numbering, and the grid axes are
// remapped by renaming %ctaid / %nctaid.
static inline uint _nv_hw_tid_x(void)      { uint v; __asm__("mov.u32 %0, %%tid.x;"  : "=r"(v)); return v; }
static inline uint _nv_hw_tid_y(void)      { uint v; __asm__("mov.u32 %0, %%tid.y;"  : "=r"(v)); return v; }
static inline uint _nv_hw_tid_z(void)      { uint v; __asm__("mov.u32 %0, %%tid.z;"  : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_x(void)     { uint v; __asm__("mov.u32 %0, %%ntid.x;" : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_y(void)     { uint v; __asm__("mov.u32 %0, %%ntid.y;" : "=r"(v)); return v; }
static inline uint _nv_hw_ntid_z(void)     { uint v; __asm__("mov.u32 %0, %%ntid.z;" : "=r"(v)); return v; }
static inline uint _nv_local_id_1(void)    { uint v; __asm__("mov.b32 %0, %%envreg27;" : "=r"(v)); return v; }
static inline uint _nv_local_id_2(void)    { uint v; __asm__("mov.b32 %0, %%envreg28;" : "=r"(v)); return v; }
static inline uint _nv_local_size_1(void)  { uint v; __asm__("mov.b32 %0, %%envreg29;" : "=r"(v)); return v; }
static inline uint _nv_local_size_2(void)  { uint v; __asm__("mov.b32 %0, %%envreg30;" : "=r"(v)); return v; }
static inline uint _nv_relin_divisor(void) { uint v; __asm__("mov.b32 %0, %%envreg31;" : "=r"(v)); return v; }

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
