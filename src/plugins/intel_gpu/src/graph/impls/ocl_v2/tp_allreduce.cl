// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/bf16_utils.cl"

// One reduction step of an AllReduce SUM: dst = src0 + src1. The operands are
// widened to f32, added, and rounded once into the element type, so f16 and
// bf16 results are what an f32 accumulator would give. Argument order and the
// 8 elements per work item are what the coordinator's append_reduce() launches.
// dst may alias src0 (in-place collective); src1 is the coordinator's staging.

#if defined(TP_BF16)
#    define ELEM_TYPE ushort
#    define LOAD8(i, p) _convert_as_bfloat168_float8(vload8(i, p))
#    define STORE8(v, i, p) vstore8(_convert_bfloat168_as_ushort8(v), i, p)
#    define LOAD1(x) _convert_as_bfloat16_float(x)
#    define STORE1(v) _convert_bfloat16_as_ushort(v)
#elif defined(TP_F16)
#    define ELEM_TYPE half
#    define LOAD8(i, p) convert_float8(vload8(i, p))
#    define STORE8(v, i, p) vstore8(convert_half8_rte(v), i, p)
#    define LOAD1(x) convert_float(x)
#    define STORE1(v) convert_half_rte(v)
#else
#    define ELEM_TYPE float
#    define LOAD8(i, p) vload8(i, p)
#    define STORE8(v, i, p) vstore8(v, i, p)
#    define LOAD1(x) (x)
#    define STORE1(v) (v)
#endif

KERNEL(tp_allreduce)(__global ELEM_TYPE* dst,
                     const __global ELEM_TYPE* src0,
                     const __global ELEM_TYPE* restrict src1,
                     ulong n) {
    const ulong i = get_global_id(0);
    const ulong base = i * 8;
    if (base + 8 <= n) {
        STORE8(LOAD8(i, src0) + LOAD8(i, src1), i, dst);
    } else if (base < n) {
        for (ulong k = base; k < n; ++k) {
            dst[k] = STORE1(LOAD1(src0[k]) + LOAD1(src1[k]));
        }
    }
}

#undef ELEM_TYPE
#undef LOAD8
#undef STORE8
#undef LOAD1
#undef STORE1
