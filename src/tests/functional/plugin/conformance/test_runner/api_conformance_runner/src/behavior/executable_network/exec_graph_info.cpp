// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <exec_graph_info.hpp>
#include "behavior/executable_network/exec_graph_info.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(ie_executable_network, ExecGraphSerializationTest,
                                ::testing::Values(ov::test::conformance::targetDevice),
                        ExecGraphSerializationTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> execGraphInfoElemTypes = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(ie_executable_network, ExecGraphUniqueNodeNames,
        ::testing::Combine(
        ::testing::ValuesIn(execGraphInfoElemTypes),
        ::testing::Values(InferenceEngine::SizeVector({1, 2, 5, 5})),
        ::testing::Values(ov::test::conformance::targetDevice)),
        ExecGraphUniqueNodeNames::getTestCaseName);

}  // namespace

