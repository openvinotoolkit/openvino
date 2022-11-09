// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;
namespace {

const std::vector<ov::AnyMap> device_modes {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"},
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}
};

const std::vector<ov::AnyMap> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
     {"GNA_SCALE_FACTOR_0", "1"},
     {"GNA_SCALE_FACTOR_1", "1"},
     {"GNA_SCALE_FACTOR_2", "1"},
     {"GNA_SCALE_FACTOR_3", "1"}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        OVInferenceChaining::getTestCaseName);
}  // namespace