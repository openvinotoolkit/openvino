// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/bf16_utils.cl"

// One reduction step of an AllReduce SUM: dst = src0 + src1. The operands are
// widened to f32, added, and rounded once into the element type, so f16 and
// bf16 results are what an f32 accumulator would give. Argument order and the
// 8 elements per work item are what the recorder's append_reduce() launches.
// dst may alias src0 (in-place collective); src1 is the collective's staging.

#define LOAD8(i, p) convert_float8(DECODE_INPUT0_COMPUTE_VECTOR_TYPE(vload8(i, p), 8))
#define LOAD1(x) convert_float(DECODE_INPUT0_COMPUTE_TYPE(x))
#if OUTPUT_IS_BF16
#    define STORE8(v, i, p) vstore8(_convert_bfloat168_as_ushort8(v), i, p)
#else
#    define STORE8(v, i, p) vstore8(CAT(convert_, MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8))(v), i, p)
#endif

KERNEL(tp_allreduce)(__global OUTPUT_TYPE* dst,
                     const __global INPUT0_TYPE* src0,
                     const __global INPUT0_TYPE* restrict src1,
                     ulong n) {
    const ulong i = get_global_id(0);
    const ulong base = i * 8;
    if (base + 8 <= n) {
        STORE8(LOAD8(i, src0) + LOAD8(i, src1), i, dst);
    } else if (base < n) {
        for (ulong k = base; k < n; ++k) {
            dst[k] = TO_OUTPUT_TYPE(LOAD1(src0[k]) + LOAD1(src1[k]));
        }
    }
}

#undef LOAD8
#undef LOAD1
#undef STORE8
