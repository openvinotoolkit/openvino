// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_tests.hpp"

gemm_base_params gemm_smoke_cases[] = {
    case8, case16, case24, case32,
    case47
};

INSTANTIATE_TEST_CASE_P(smoke_GPU_GemmRandomTest, GemmRandomTest,
    testing::Combine(
        testing::Values("GPU"),
        testing::Values("FP32", "FP16"),
        testing::ValuesIn(gemm_smoke_cases)
));

gemm_base_params gemm_all_cases[] = {
    case1,  case2,  case3,  case4,  case5,  case6,  case7,
    case9,  case10, case11, case12, case13, case14, case15,
    case17, case18, case19, case20, case21, case22, case23,
    case25, case26, case27, case28, case29, case30, case31,
    case33, case34, case35, case36, case37, case38,
    case39, case40, case41, case42, case43, case44,
    case45, case46
};

INSTANTIATE_TEST_CASE_P(nightly_GPU_GemmRandomTest, GemmRandomTest,
    testing::Combine(
        testing::Values("GPU"),
        testing::Values("FP32", "FP16"),
        testing::ValuesIn(gemm_all_cases)
));
