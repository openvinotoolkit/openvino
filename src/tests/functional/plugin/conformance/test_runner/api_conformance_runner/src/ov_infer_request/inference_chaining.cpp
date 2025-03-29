// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference_chaining.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ov_api_conformance_helpers.hpp"

namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferenceChainingStatic,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVInferenceChaining::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_infer_request, OVInferenceChaining,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVInferenceChaining::getTestCaseName);
}  // namespace
