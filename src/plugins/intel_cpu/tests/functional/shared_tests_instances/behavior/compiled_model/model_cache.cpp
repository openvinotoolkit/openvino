// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/model_cache.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;

INSTANTIATE_TEST_SUITE_P(smoke_,
                         WeightlessCacheAccuracy,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(model_dtypes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         WeightlessCacheAccuracy::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_,
                         WeightlessCacheAccuracyLowPrecision,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(inference_modes),
                                            ::testing::ValuesIn(low_precision_dtypes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         WeightlessCacheAccuracy::get_test_case_name);
