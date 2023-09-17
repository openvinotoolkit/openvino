// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::behavior;
namespace {

// GNA_SW_EXACT mode excluded from the tests because without quantization we got the bad accuracy for this test
const std::vector<ov::AnyMap> device_modes{{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferenceChaining,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(device_modes)),
                         OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferenceChainingStatic,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(device_modes)),
                         OVInferenceChainingStatic::getTestCaseName);
}  // namespace
