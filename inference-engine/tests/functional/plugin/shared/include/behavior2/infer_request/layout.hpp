// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/behavior_test_utils.hpp"

namespace BehaviorTestsDefinitions {
typedef std::tuple<
        InferenceEngine::Precision,             // Network precision
        std::string,                            // Target device name
        std::map<std::string, std::string>,     // Target config
        InferenceEngine::Layout,                // Layout
        std::vector<size_t>>                    // InputShapes
LayoutParams;

class InferRequestLayoutTest : public CommonTestUtils::TestsCommon,
                   public ::testing::WithParamInterface<LayoutParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayoutParams> obj);
    void SetUp() override;
    void TearDown() override;

protected:
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    InferenceEngine::Layout layout;
    std::vector<size_t> inputShapes;
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
};
}  // namespace BehaviorTestsDefinitions