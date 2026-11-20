// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define DQ_MAX_SEARCH_INIT_VAL 0.000000059604645h

#define DQ_TO_TYPE_N_(type, n, x) convert_##type##n(x)
#define DQ_TO_TYPE_N(type, n, x) DQ_TO_TYPE_N_(type, n, x)
#define DQ_TO_TYPE_SAT_(type, x) _convert_##type##_sat(x)
#define DQ_TO_TYPE_SAT(type, x) DQ_TO_TYPE_SAT_(type, x)
#define DQ_TO_TYPE_N_SAT_(type, n, x) _convert_##type##n##_sat(x)
#define DQ_TO_TYPE_N_SAT(type, n, x) DQ_TO_TYPE_N_SAT_(type, n, x)

#define DQ_COMPUTE_MXFP_SCALE(x) (exp2(floor(log2(_convert_float(OUTPUT_VAL_MAX) / x))))
#define DQ_COMPUTE_OUTPUT_SCALE(x) (TO_OUTPUT1_TYPE(1.0f / x))
#define DQ_COMPUTE_OUTPUT_VALUE(v, s) (DQ_TO_TYPE_SAT(OUTPUT_TYPE, v * s))
#define DQ_COMPUTE_OUTPUT_VALUE_N(n, v, s) (DQ_TO_TYPE_N_SAT(OUTPUT_TYPE, n, DQ_TO_TYPE_N(float, n, v) * (MAKE_VECTOR_TYPE(float, n))s))
#define DQ_CAST_TO_OUTPUT_TYPE(x) (DQ_TO_TYPE_SAT(OUTPUT_TYPE, x))
#define DQ_CAST_TO_OUTPUT_TYPE_N(n, x) (DQ_TO_TYPE_N_SAT(OUTPUT_TYPE, n, x))

// Symmetric int8 quantization, shared by dynamic_quantize_gpu_opt and the activations of the MoE expert GEMMs
// (moe_3gemm_dyn_quan): a group of values v is stored as q = rte(v * s) with s = 127 / max(|v|), computed in the type of
// v, and dequantized as q / s. The sum of q over the group (the precomputed reduction, DQ_SUM_N) lets a GEMM apply a
// weight zero point once per group.
#define DQ_I8_MAX_VAL 127
// Lower bound of max(|v|), so that 127 / max(|v|) stays finite in half
#define DQ_I8_MIN_ABS_MAX 0.003h
#define DQ_I8_SCALE(type, abs_max) ((type)DQ_I8_MAX_VAL / (abs_max))
#define DQ_I8_QUANTIZE_N(n, v, scale) CAT(CAT(convert_char, n), _rte)((v) * (scale))

// Statements: fold the n components of vector v into abs_max, and add the n components of vector q to sum
#define DQ_ABS_MAX_N(n, v, abs_max)                    \
    unroll_for (uint dq_i = 0; dq_i < (n); ++dq_i) {   \
        abs_max = fmax(abs_max, fabs((v)[dq_i]));      \
    }
#define DQ_SUM_N(n, q, sum)                            \
    unroll_for (uint dq_i = 0; dq_i < (n); ++dq_i) {   \
        sum += (q)[dq_i];                              \
    }

// One group spread over a subgroup, each lane holding n values v (a vector of type): q receives the quantized values of
// the lane and dq_scale the dequantization scale 1 / s, uniform in the subgroup.
#define DQ_I8_SUBGROUP_QUANTIZE_N(n, type, v, q, dq_scale)                              \
    do {                                                                               \
        type dq_abs_max_ = DQ_I8_MIN_ABS_MAX;                                          \
        DQ_ABS_MAX_N(n, v, dq_abs_max_)                                                \
        const type dq_s_ = DQ_I8_SCALE(type, sub_group_reduce_max(dq_abs_max_));       \
        q = DQ_I8_QUANTIZE_N(n, v, dq_s_);                                             \
        dq_scale = (type)1 / dq_s_;                                                    \
    } while (0)
