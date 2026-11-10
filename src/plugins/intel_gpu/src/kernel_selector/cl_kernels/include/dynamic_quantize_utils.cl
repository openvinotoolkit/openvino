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
