// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Optimized broadcast elementwise multiply for FLUX transformer hot paths (AdaLayerNorm, RMSNorm,
// SwiGLU, scalar activation). Vectorized half8, LOCAL_SIZE=512 tuned on Battlemage; runtime total
// and period support dynamic shapes. out[i] = a[i] * b[i % period].

#if FULL_IS_INPUT0
#    define A_PTR input0
#    define B_PTR input1
#    define A_OFFSET INPUT0_OFFSET
#    define B_OFFSET INPUT1_OFFSET
#else
#    define A_PTR input1
#    define B_PTR input0
#    define A_OFFSET INPUT1_OFFSET
#    define B_OFFSET INPUT0_OFFSET
#endif

#define VEC_SIZE 8
#define LOCAL_SIZE 512

#define FLUX_PERIOD 3072

#define WRAP_IDX(c, period, k) (((c) + (k) < (period)) ? ((c) + (k)) : ((c) + (k) - (period)))

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__attribute__((vec_type_hint(half8)))
KERNEL(eltwise_broadcast_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict input0,
    const __global INPUT1_TYPE* restrict input1,
    __global OUTPUT_TYPE* restrict output,
    const uint total,
    const uint period)
{
    const int gid = get_global_id(0);
    const int base = gid * VEC_SIZE;
    const int remaining = (int)total - base;

    if (remaining <= 0) {
        return;
    }

    prefetch(A_PTR + A_OFFSET + base + VEC_SIZE * LOCAL_SIZE, VEC_SIZE);

    if (remaining >= VEC_SIZE) {
        const half8 va = vload8(0, A_PTR + A_OFFSET + base);
        half8 vb;

        if (period == 1u) {
            half s_lane = (half)0;
            if (get_sub_group_local_id() == 0u) {
                s_lane = B_PTR[B_OFFSET];
            }
            const half s = sub_group_broadcast(s_lane, 0u);
            vb = (half8)(s, s, s, s, s, s, s, s);

        } else if (period == total) {
            vb = vload8(0, B_PTR + B_OFFSET + base);

        } else if (period == (uint)FLUX_PERIOD) {
            const int c = base % FLUX_PERIOD;
            if (c + VEC_SIZE <= FLUX_PERIOD) {
                vb = vload8(0, B_PTR + B_OFFSET + c);
            } else {
                vb = (half8)(
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 0)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 1)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 2)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 3)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 4)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 5)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 6)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, FLUX_PERIOD, 7)]
                );
            }

        } else if (period == 2u) {
            const half2 p = vload2(0, B_PTR + B_OFFSET);
            vb = (half8)(p.s0, p.s1, p.s0, p.s1, p.s0, p.s1, p.s0, p.s1);

        } else if (period == 4u) {
            const half4 p = vload4(0, B_PTR + B_OFFSET);
            vb = (half8)(p.s0, p.s1, p.s2, p.s3, p.s0, p.s1, p.s2, p.s3);

        } else if (period == 8u) {
            vb = vload8(0, B_PTR + B_OFFSET);

        } else {
            int c;
            if ((period & (period - 1u)) == 0u) {
                c = base & (int)(period - 1u);
            } else {
                c = base % (int)period;
            }

            if (c + VEC_SIZE <= (int)period) {
                vb = vload8(0, B_PTR + B_OFFSET + c);

            } else if (period >= (uint)VEC_SIZE) {
                vb = (half8)(
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 0)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 1)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 2)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 3)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 4)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 5)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 6)],
                    B_PTR[B_OFFSET + WRAP_IDX(c, period, 7)]
                );

            } else {
                vb = (half8)(
                    B_PTR[B_OFFSET + ((c + 0) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 1) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 2) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 3) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 4) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 5) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 6) % (int)period)],
                    B_PTR[B_OFFSET + ((c + 7) % (int)period)]
                );
            }
        }

        vstore8(va * vb, 0, output + OUTPUT_OFFSET + base);
        return;
    }

#pragma unroll
    for (int k = 0; k < VEC_SIZE; ++k) {
        const int i = base + k;
        if (i < (int)total) {
            int bi;
            if (period == 1u) {
                bi = 0;
            } else if (period == total) {
                bi = i;
            } else if ((period & (period - 1u)) == 0u) {
                bi = i & (int)(period - 1u);
            } else {
                bi = i % (int)period;
            }
            output[OUTPUT_OFFSET + i] = A_PTR[A_OFFSET + i] * B_PTR[B_OFFSET + bi];
        }
    }
}

#undef WRAP_IDX
#undef FLUX_PERIOD
#undef LOCAL_SIZE
#undef VEC_SIZE
#undef A_PTR
#undef B_PTR
#undef A_OFFSET
#undef B_OFFSET
