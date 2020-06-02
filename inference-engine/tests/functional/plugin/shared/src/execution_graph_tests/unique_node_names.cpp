// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "execution_graph_tests/unique_node_names.hpp"

#include "network_serializer.h"

namespace LayerTestsDefinitions {

std::string ExecGraphUniqueNodeNames::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(inputPrecision, netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ExecGraphUniqueNodeNames::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision inputPrecision, netPrecision;
    std::tie(inputPrecision, netPrecision, inputShape, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto concat = std::make_shared<ngraph::opset1::Concat>(split->outputs(), 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
}

TEST_P(ExecGraphUniqueNodeNames, CheckUniqueNodeNames) {
    InferenceEngine::CNNNetwork cnnNet(fnPtr);

    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);

    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    auto nodes = InferenceEngine::Serialization::TopologicalSort(execGraphInfo);

    int numReorders = 0;
    int expectedReorders = 2;
    std::unordered_set<std::string> names;
    for (auto &node : nodes) {
        IE_SUPPRESS_DEPRECATED_START
        ASSERT_TRUE(names.find(node->name) == names.end()) << "Node with name " << node->name << "already exists";
        names.insert(node->name);
        if (node->type == "Reorder") {
            numReorders++;
        }
        IE_SUPPRESS_DEPRECATED_END
    }
    ASSERT_TRUE(numReorders == expectedReorders) << "Expected reorders: " << expectedReorders << ", actual reorders: " << numReorders;

    fnPtr.reset();
};

}  // namespace LayerTestsDefinitions
