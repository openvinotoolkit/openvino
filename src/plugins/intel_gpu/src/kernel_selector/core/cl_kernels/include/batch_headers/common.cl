// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset) ((__global elem_type*)((__global char*)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset) ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#   define ASSUME_HINT(x) __builtin_assume(x)
#else
#   define ASSUME_HINT(x) do { } while (0)
#endif
