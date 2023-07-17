// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "common_test_utils/test_common.hpp"
#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
typedef std::tuple<
        InferenceEngine::CNNNetwork, // CNNNetwork to work with
        std::vector<std::string>,    // Memory States to query
        std::string,               // Target device name
        std::map<std::string, std::string>> // device configuration
memoryStateParams;

class InferRequestVariableStateTest : public BehaviorTestsUtils::IEInferRequestTestBase,
                                      public testing::WithParamInterface<memoryStateParams> {
protected:
    InferenceEngine::CNNNetwork net;
    std::vector<std::string> statesToQuery;
    std::string deviceName;
    std::map<std::string, std::string> configuration;
    InferenceEngine::ExecutableNetwork PrepareNetwork();

public:
    void SetUp() override;
    static std::string getTestCaseName(const testing::TestParamInfo<memoryStateParams> &obj);
    static InferenceEngine::CNNNetwork getNetwork();
};

using InferRequestQueryStateExceptionTest = InferRequestVariableStateTest;
} // namespace BehaviorTestsDefinitions
