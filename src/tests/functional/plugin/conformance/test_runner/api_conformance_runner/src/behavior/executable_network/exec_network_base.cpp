// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

#include "api_conformance_helpers.hpp"

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

namespace {
    INSTANTIATE_TEST_SUITE_P(ie_executable_network, ExecutableNetworkBaseTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(return_all_possible_device_combination()),
                                    ::testing::Values(ie_config)),
                            ExecutableNetworkBaseTest::getTestCaseName);

    const std::vector<InferenceEngine::Precision> execNetBaseElemTypes = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::FP64,
            InferenceEngine::Precision::BF16,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::U16,
            InferenceEngine::Precision::U32,
            InferenceEngine::Precision::U64,
            InferenceEngine::Precision::I8,
            InferenceEngine::Precision::I16,
            InferenceEngine::Precision::I32,
            InferenceEngine::Precision::I64,
            InferenceEngine::Precision::BOOL,
    };

    INSTANTIATE_TEST_SUITE_P(ie_executable_network, ExecNetSetPrecision,
                            ::testing::Combine(
                                    ::testing::ValuesIn(execNetBaseElemTypes),
                                    ::testing::ValuesIn(return_all_possible_device_combination()),
                                    ::testing::Values(ie_config)),
                            ExecNetSetPrecision::getTestCaseName);
}  // namespace
