// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_tests.hpp"

quantize_test_params quantize_only_test_cases[] = {
        quantize_test_params{"GPU", case_1},
        quantize_test_params{"GPU", case_2},
        quantize_test_params{"GPU", case_3},
        quantize_test_params{"GPU", case_4},
        quantize_test_params{"GPU", case_5},
        quantize_test_params{"GPU", case_6},
        quantize_test_params{"GPU", case_7},
        quantize_test_params{"GPU", case_8},
        quantize_test_params{"GPU", case_9},
        quantize_test_params{"GPU", case_10},
        quantize_test_params{"GPU", case_11},
        quantize_test_params{"GPU", case_12},
        quantize_test_params{"GPU", case_13},
        quantize_test_params{"GPU", case_14},
        quantize_test_params{"GPU", case_15},
        quantize_test_params{"GPU", case_16},
        quantize_test_params{"GPU", case_17},
        quantize_test_params{"GPU", case_18},
        quantize_test_params{"GPU", case_19},
        quantize_test_params{"GPU", case_20},
        quantize_test_params{"GPU", case_21},
        quantize_test_params{"GPU", case_22},
        quantize_test_params{"GPU", case_23},
        quantize_test_params{"GPU", case_24},
};

INSTANTIATE_TEST_CASE_P(smoke_GPU_TestQuantize, QuantizeOnlyTest, ::testing::ValuesIn(quantize_only_test_cases));
