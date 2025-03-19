// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"

#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/add_output.hpp"

std::string AddOutputsTest::getTestCaseName(const testing::TestParamInfo<addOutputsParams> &obj) {
    std::ostringstream results;
    std::shared_ptr<ov::Model> net;
    std::vector<std::string> outputsToAdd;
    std::string deviceName;
    std::tie(net, outputsToAdd, deviceName) = obj.param;
    results << "Outputs=" << ov::test::utils::vec2str<std::string>(outputsToAdd);
    results << "Dev=" << deviceName;
    return results.str();
}

void AddOutputsTest::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
}

TEST_P(AddOutputsTest, smoke_CheckOutputExist) {
    std::shared_ptr<ov::Model> net;
    std::vector<std::string> outputsToAdd;
    std::string deviceName;
    std::tie(net, outputsToAdd, deviceName) = GetParam();
    std::vector<std::string> expectedOutputs = outputsToAdd;
    for (const auto &out : net->outputs()) {
        expectedOutputs.push_back(out.get_any_name());
    }

    for (const auto &out : outputsToAdd) {
        net->add_output(out);
    }
    auto ie = ov::test::utils::PluginCache::get().core(deviceName);
    auto executableNet = ie->compile_model(net, deviceName);
    auto outputs = executableNet.outputs();

    for (const auto &expected_out_name : expectedOutputs) {
        auto res = std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<const ov::Node>& out){
            return expected_out_name == out.get_any_name();
        }) != outputs.end();
        ASSERT_TRUE(res) << "Layer " << expected_out_name << " expected to be in network outputs but it's not!";
    }
}
