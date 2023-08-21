// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"

#include "ov_api_conformance_helpers.hpp"

using namespace ov::test::behavior;
using namespace ov::test::conformance;

namespace {

INSTANTIATE_TEST_SUITE_P(ov_infer_request_mandatory, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::Values(pluginConfig)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
