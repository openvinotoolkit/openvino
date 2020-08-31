// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero/query_network.hpp"
#include <ngraph/op/util/op_types.hpp>
#include <ngraph/variant.hpp>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include <random>
namespace HeteroTests {

void QueryNetworkTest::SetUp() {
    auto& param = GetParam();
    targetDevice = std::get<Plugin>(param);
    function = std::get<Function>(param);
    cnnNetwork = InferenceEngine::CNNNetwork{function};
}

std::string QueryNetworkTest::getTestCaseName(const ::testing::TestParamInfo<QueryNetworkTestParameters>& obj) {
    return "function=" + std::get<Function>(obj.param)->get_friendly_name() + "_targetDevice=" + std::get<Plugin>(obj.param);
}

TEST_P(QueryNetworkTest, queryNetworkResultContainAllAndOnlyInputLayers) {
    auto& param = GetParam();
    auto queryNetworkResult = PluginCache::get().ie()->QueryNetwork(cnnNetwork, std::get<Plugin>(param));
    ASSERT_NE(nullptr, cnnNetwork.getFunction());
    std::set<std::string> expectedLayers;
    for (auto&& node : function->get_ops()) {
        expectedLayers.insert(node->get_friendly_name());
    }
    std::set<std::string> actualLayers;
    for (auto&& res : queryNetworkResult.supportedLayersMap) {
        actualLayers.insert(res.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}
}  //  namespace HeteroTests
