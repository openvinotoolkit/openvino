// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include <exec_graph_info.hpp>
#include "behavior/executable_network/exec_graph_info.hpp"

namespace {

using namespace ExecutionGraphTests;

INSTANTIATE_TEST_SUITE_P(smoke_serialization, ExecGraphSerializationTest,
                                ::testing::Values(ov::test::utils::DEVICE_CPU),
                        ExecGraphSerializationTest::getTestCaseName);

TEST_P(ExecGraphUniqueNodeNames, CheckUniqueNodeNames) {
InferenceEngine::CNNNetwork cnnNet(fnPtr);

auto ie = PluginCache::get().ie();
auto execNet = ie->LoadNetwork(cnnNet, target_device);

InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();

int numReorders = 0;
int expectedReorders = 2;
std::unordered_set<std::string> names;

auto function = execGraphInfo.getFunction();
ASSERT_NE(function, nullptr);

for (const auto & op : function->get_ops()) {
const auto & rtInfo = op->get_rt_info();
auto it = rtInfo.find(ExecGraphInfoSerialization::LAYER_TYPE);
ASSERT_NE(rtInfo.end(), it);
auto opType = it->second.as<std::string>();

if (opType == "Reorder") {
numReorders++;
}
}

ASSERT_EQ(numReorders, expectedReorders) << "Expected reorders: " << expectedReorders << ", actual reorders: " << numReorders;
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ExecGraphUniqueNodeNames,
        ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({1, 2, 5, 5})),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ExecGraphUniqueNodeNames::getTestCaseName);

}  // namespace

