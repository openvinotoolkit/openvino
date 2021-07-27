// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "common_test_utils/test_common.hpp"
#include <ie_core.hpp>

namespace BehaviorTestsDefinitions {
typedef std::tuple<
        InferenceEngine::CNNNetwork, // CNNNetwork to work with
        std::vector<std::string>,    // Memory States to query
        std::string>                 // Target device name
memoryStateParams;

class InferRequestVariableStateTest : public CommonTestUtils::TestsCommon,
                                      public testing::WithParamInterface<memoryStateParams> {
protected:
    InferenceEngine::CNNNetwork net;
    std::vector<std::string> statesToQuery;
    std::string deviceName;

    void SetUp() override;

    InferenceEngine::ExecutableNetwork PrepareNetwork();

public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryStateParams> &obj);
};
} // namespace BehaviorTestsDefinitions
