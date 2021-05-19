// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/common_utils.hpp>
#include "behavior/add_output.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

std::string AddOutputsTest::getTestCaseName(const testing::TestParamInfo<addOutputsParams> &obj) {
    std::ostringstream results;
    InferenceEngine::CNNNetwork net;
    std::vector<std::string> outputsToAdd;
    std::string deviceName;
    std::tie(net, outputsToAdd, deviceName) = obj.param;
    results << "Outputs:" << CommonTestUtils::vec2str<std::string>(outputsToAdd);
    return results.str();
}

void AddOutputsTest::SetUp() {
    std::tie(net, outputsToAdd, deviceName) = GetParam();
}

TEST_P(AddOutputsTest, smoke_CheckOutputExist) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::vector<std::string> expectedOutputs = outputsToAdd;
    for (const auto &out : net.getOutputsInfo()) {
        expectedOutputs.push_back(out.first);
    }
    for (const auto &out : outputsToAdd) {
        net.addOutput(out);
    }
    auto ie = PluginCache::get().ie(deviceName);
    auto executableNet = ie->LoadNetwork(net, deviceName);
    auto outputs = executableNet.GetOutputsInfo();

    for (const auto &out : expectedOutputs) {
        ASSERT_TRUE(outputs.count(out)) << "Layer " << out << " expected to be in network outputs but it's not!";
    }
}
