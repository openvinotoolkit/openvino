// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <unordered_set>
#include <string>
#include <functional>

#include <ie_core.hpp>
#include <ngraph/function.hpp>
#include <exec_graph_info.hpp>
#include <ngraph/variant.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/unique_node_names.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphUniqueNodeNames::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ExecGraphUniqueNodeNames::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);
    auto concat = std::make_shared<ngraph::opset1::Concat>(split->outputs(), 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
}

void ExecGraphUniqueNodeNames::TearDown() {
}

TEST_P(ExecGraphUniqueNodeNames, CheckUniqueNodeNames) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::CNNNetwork cnnNet(fnPtr);

    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);

    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();

    int numReorders = 0;
    int expectedReorders = 2;
    std::unordered_set<std::string> names;

    auto function = execGraphInfo.getFunction();
    ASSERT_NE(function, nullptr);

    for (const auto & op : function->get_ops()) {
        ASSERT_TRUE(names.find(op->get_friendly_name()) == names.end()) <<
            "Node with name " << op->get_friendly_name() << "already exists";
        names.insert(op->get_friendly_name());

        const auto & rtInfo = op->get_rt_info();
        auto it = rtInfo.find(ExecGraphInfoSerialization::LAYER_TYPE);
        ASSERT_NE(rtInfo.end(), it);
        auto opType = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
        ASSERT_NE(nullptr, opType);

        if (opType->get() == "Reorder") {
            numReorders++;
        }
    }

    if (targetDevice != "GPU" )
        ASSERT_TRUE(numReorders == expectedReorders) << "Expected reorders: " << expectedReorders << ", actual reorders: " << numReorders;

    fnPtr.reset();
};

}  // namespace ExecutionGraphTests
