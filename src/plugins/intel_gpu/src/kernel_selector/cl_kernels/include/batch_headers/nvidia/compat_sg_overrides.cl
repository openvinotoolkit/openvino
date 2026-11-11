// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//
// Binds the sub-group helpers that the shared batch headers define as functions to the parameterised implementations
// in nvidia/compat_sg.cl.
//
// Without cl_intel_subgroups, sub_group_block_read.cl, sub_group_block_write.cl and sub_group_shuffle.cl define
// _sub_group_block_read*, _sub_group_block_write* and _sub_group_shuffle as functions, which keep the 32 default of
// _NV_SUB_GROUP_SIZE whatever a kernel asks for. Function-like macros of the same names shadow the call sites instead,
// leaving the shared headers - and the Intel builds - untouched; the 32-wide definitions go unreferenced.
// kernels_cache.cpp therefore places this file after the batch headers and before the kernels: any earlier, it would
// macro-expand the names inside their own declarations.

#ifdef __NV_CL_C_VERSION

// compat_common.cl defines cl_intel_subgroups_short, so the batch headers define the ushort accessors as macros; drop
// them first, so the definitions below replace them rather than redefine them.
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
