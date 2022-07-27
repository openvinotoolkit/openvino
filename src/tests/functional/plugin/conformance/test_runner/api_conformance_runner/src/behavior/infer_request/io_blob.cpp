// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

INSTANTIATE_TEST_SUITE_P(ie_infer_request, InferRequestIOBBlobTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::ValuesIn(empty_config)),
                         InferRequestIOBBlobTest::getTestCaseName);
}  // namespace
