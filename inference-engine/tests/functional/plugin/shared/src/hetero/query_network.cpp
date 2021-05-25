// Copyright (C) 2018-2021 Intel Corporation
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
    std::pair<std::set<std::string>, std::shared_ptr<ngraph::Function>> graphAndLayers;
    graphAndLayers = std::get<Function>(param);
    expectedLayers = graphAndLayers.first;
    function = graphAndLayers.second;
    cnnNetwork = InferenceEngine::CNNNetwork{function};
}

std::string QueryNetworkTest::getTestCaseName(const ::testing::TestParamInfo<QueryNetworkTestParameters>& obj) {
    return "function=" + std::get<Function>(obj.param).second->get_friendly_name() + "_targetDevice=" + std::get<Plugin>(obj.param);
}

std::pair<std::set<std::string>, std::shared_ptr<ngraph::Function>> QueryNetworkTest::generateParams(std::shared_ptr<ngraph::Function> graph) {
    std::set<std::string> layers;
    for (auto&& node : graph->get_ops()) {
        layers.insert(node->get_friendly_name());
    }
    return std::make_pair(layers, graph);
}

TEST_P(QueryNetworkTest, queryNetworkResultContainAllAndOnlyInputLayers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto& param = GetParam();
    auto queryNetworkResult = PluginCache::get().ie()->QueryNetwork(cnnNetwork, std::get<Plugin>(param));
    ASSERT_NE(nullptr, cnnNetwork.getFunction());

    std::set<std::string> actualLayers;
    for (auto&& res : queryNetworkResult.supportedLayersMap) {
        actualLayers.insert(res.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

}  //  namespace HeteroTests
