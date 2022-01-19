// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/add_output.hpp"

std::string AddOutputsTest::getTestCaseName(const testing::TestParamInfo<addOutputsParams> &obj) {
    std::ostringstream results;
    std::vector<std::string> outputsToAdd;
    std::tie(std::ignore, outputsToAdd, std::ignore) = obj.param;
    results << "Outputs:" << CommonTestUtils::vec2str<std::string>(outputsToAdd);
    return results.str();
}

void AddOutputsTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(createNetwork, outputsToAdd, deviceName) = GetParam();
}

TEST_P(AddOutputsTest, smoke_CheckOutputExist) {
    std::vector<std::string> expectedOutputs = outputsToAdd;
    auto net = createNetwork();
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
