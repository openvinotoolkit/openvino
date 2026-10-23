// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Binds the sub-group helpers that the shared batch headers define as plain functions to the parameterised
// implementations in nvidia/compat_sg.cl.
//
// Why this lives in a separate file instead of the prelude: sub_group_block_read.cl, sub_group_block_write.cl and
// sub_group_shuffle.cl declare _sub_group_block_read*, _sub_group_block_write* and _sub_group_shuffle as functions
// whenever cl_intel_subgroups is unavailable and __OPENCL_C_VERSION__ >= 200 - which is exactly the NVIDIA case.
// Those bodies call get_sub_group_local_id() / get_max_sub_group_size() / sub_group_broadcast(), so they capture
// whatever _NV_SUB_GROUP_SIZE happens to be where the header is expanded. The batch headers are emitted once at the
// top of the program, so that is always the 32 default, and redefining _NV_SUB_GROUP_SIZE per kernel afterwards has
// no effect on them.
//
// Defining these names as function-like macros shadows the call sites without disturbing the declarations, which is
// legal C and keeps the shared headers untouched - and therefore keeps Intel builds byte-identical. The 32-wide
// definitions the headers emitted stay in the translation unit but are inline and unreferenced, so they are dropped.
//
// Ordering is load-bearing: kernels_cache.cpp splices this file *after* the batch headers and *before* the kernel
// sources. Placing it earlier would macro-expand the function names inside their own declarations.

#ifdef __NV_CL_C_VERSION

// The prelude defines cl_intel_subgroups_short, so sub_group_block_read.cl and sub_group_block_write.cl spell the
// ushort accessors as macros forwarding to intel_sub_group_block_*_us* instead of declaring functions. Drop those
// macros first, so that the definitions below replace them rather than redefine them: C does not allow a function-like
// macro to be redefined with a different replacement list, and the NVIDIA compiler warns about each of the eight names.
#undef _sub_group_block_read_us
#undef _sub_group_block_read_us2
#undef _sub_group_block_read_us4
#undef _sub_group_block_read_us8
#undef _sub_group_block_write_us
#undef _sub_group_block_write_us2
#undef _sub_group_block_write_us4
#undef _sub_group_block_write_us8

// __global variants of the block accessors; the _slm spellings below reach the same helpers through their __local
// overloads.
#define _sub_group_block_read(ptr)          _nv_sub_group_block_read((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read2(ptr)         _nv_sub_group_block_read2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read4(ptr)         _nv_sub_group_block_read4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read8(ptr)         _nv_sub_group_block_read8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_us(ptr)       _nv_sub_group_block_read_us((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_us2(ptr)      _nv_sub_group_block_read_us2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_us4(ptr)      _nv_sub_group_block_read_us4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_us8(ptr)      _nv_sub_group_block_read_us8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_uc(ptr)       _nv_sub_group_block_read_uc((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_uc2(ptr)      _nv_sub_group_block_read_uc2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_uc4(ptr)      _nv_sub_group_block_read_uc4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_uc8(ptr)      _nv_sub_group_block_read_uc8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_uc16(ptr)     _nv_sub_group_block_read_uc16((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_ul(ptr)       _nv_sub_group_block_read_ul((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_ul2(ptr)      _nv_sub_group_block_read_ul2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_ul4(ptr)      _nv_sub_group_block_read_ul4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_ul8(ptr)      _nv_sub_group_block_read_ul8((ptr), _NV_SUB_GROUP_SIZE)

#define _sub_group_block_read_slm(ptr)      _nv_sub_group_block_read((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm2(ptr)     _nv_sub_group_block_read2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm4(ptr)     _nv_sub_group_block_read4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm8(ptr)     _nv_sub_group_block_read8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_us(ptr)   _nv_sub_group_block_read_us((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_us2(ptr)  _nv_sub_group_block_read_us2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_us4(ptr)  _nv_sub_group_block_read_us4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_us8(ptr)  _nv_sub_group_block_read_us8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_uc(ptr)   _nv_sub_group_block_read_uc((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_uc2(ptr)  _nv_sub_group_block_read_uc2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_uc4(ptr)  _nv_sub_group_block_read_uc4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_uc8(ptr)  _nv_sub_group_block_read_uc8((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_uc16(ptr) _nv_sub_group_block_read_uc16((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_ul(ptr)   _nv_sub_group_block_read_ul((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_ul2(ptr)  _nv_sub_group_block_read_ul2((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_ul4(ptr)  _nv_sub_group_block_read_ul4((ptr), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_read_slm_ul8(ptr)  _nv_sub_group_block_read_ul8((ptr), _NV_SUB_GROUP_SIZE)

// BLOCK_WRITEN_SLM routes __local pointers through the same names as BLOCK_WRITEN, which the emulated declarations in
// sub_group_block_write.cl cannot accept because they are typed __global. The __local overload fixes that too.
#define _sub_group_block_write(ptr, v)      _nv_sub_group_block_write((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write2(ptr, v)     _nv_sub_group_block_write2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write4(ptr, v)     _nv_sub_group_block_write4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write8(ptr, v)     _nv_sub_group_block_write8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_us(ptr, v)   _nv_sub_group_block_write_us((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_us2(ptr, v)  _nv_sub_group_block_write_us2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_us4(ptr, v)  _nv_sub_group_block_write_us4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_us8(ptr, v)  _nv_sub_group_block_write_us8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_uc(ptr, v)   _nv_sub_group_block_write_uc((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_uc2(ptr, v)  _nv_sub_group_block_write_uc2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_uc4(ptr, v)  _nv_sub_group_block_write_uc4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_uc8(ptr, v)  _nv_sub_group_block_write_uc8((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_uc16(ptr, v) _nv_sub_group_block_write_uc16((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_ul(ptr, v)   _nv_sub_group_block_write_ul((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_ul2(ptr, v)  _nv_sub_group_block_write_ul2((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_ul4(ptr, v)  _nv_sub_group_block_write_ul4((ptr), (v), _NV_SUB_GROUP_SIZE)
#define _sub_group_block_write_ul8(ptr, v)  _nv_sub_group_block_write_ul8((ptr), (v), _NV_SUB_GROUP_SIZE)

// sub_group_shuffle.cl declares _sub_group_shuffle on this path but has no _sub_group_shuffle_up / _down at all, so
// these two exist only here.
#define _sub_group_shuffle(v, c)                  _nv_sub_group_shuffle((v), (c), _NV_SUB_GROUP_SIZE)
#define _sub_group_shuffle_up(prev, cur, delta)   _nv_sub_group_shuffle_up((prev), (cur), (delta), _NV_SUB_GROUP_SIZE)
#define _sub_group_shuffle_down(cur, next, delta) _nv_sub_group_shuffle_down((cur), (next), (delta), _NV_SUB_GROUP_SIZE)

#endif // __NV_CL_C_VERSION
