// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/wait.hpp"

#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferRequestWaitTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                            OVInferRequestWaitTests::getTestCaseName);

}  // namespace
