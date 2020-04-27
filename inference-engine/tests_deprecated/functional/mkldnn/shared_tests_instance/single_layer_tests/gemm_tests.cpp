// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_tests.hpp"

gemm_base_params gemm_smoke_cases[] = {
    case6, case14, case22, case30,
    case38
};

INSTANTIATE_TEST_CASE_P(smoke_CPU_GemmRandomTest, GemmRandomTest,
    testing::Combine(
        testing::Values("CPU"),
        testing::Values("FP32"),
        testing::ValuesIn(gemm_smoke_cases)
    ));

gemm_base_params gemm_all_cases[] = {        // 5D cases
    case1,  case2,  case3,  case4,  case5,   /* case7,  case8,  */
    case9,  case10, case11, case12, case13,  /* case15, case16, */
    case17, case18, case19, case20, case21,  /* case23, case24, */
    case25, case26, case27, case28, case29,  /* case31, case32, */
    case33, case34, case35, case36, case37, case38,
    // Cases with mismatched input dimension numbers
    // case39, case40, case41, case42, case43, case44,
    // case45, case46, case47
};

INSTANTIATE_TEST_CASE_P(nightly_CPU_GemmRandomTest, GemmRandomTest,
    testing::Combine(
        testing::Values("CPU"),
        testing::Values("FP32"),
        testing::ValuesIn(gemm_all_cases)
    ));
