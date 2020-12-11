// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include <exec_graph_info.hpp>

#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"

std::vector<InferenceEngine::CNNLayerPtr> TopologicalSort(const InferenceEngine::ICNNNetwork& network);

namespace ExecutionGraphTests {

std::string ExecGraphInputsFusingBinConv::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice = obj.param;
    return "targetDevice=" + targetDevice;
}

void ExecGraphInputsFusingBinConv::SetUp() {
    const InferenceEngine::SizeVector inputShapes = { 1, 16, 30, 30}, binConvKernelSize = {2, 2}, convKernelSize = {3, 3};
    const size_t numOutChannels = 16, numGroups = 16;
    const std::vector<size_t > strides = {1, 1}, dilations = {1, 1};
    const std::vector<ptrdiff_t> padsBegin = {1, 1}, padsEnd = {0, 0};
    const ngraph::op::PadType paddingType = ngraph::op::PadType::EXPLICIT;
    const float padValue = 1.0;
    targetDevice = this->GetParam();

    auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShapes});
    auto binConv = ngraph::builder::makeBinaryConvolution(params[0], binConvKernelSize, strides, padsBegin, padsEnd, dilations, paddingType, numOutChannels,
                                                          padValue);
    auto conv = ngraph::builder::makeGroupConvolution(binConv, ngraph::element::f32, convKernelSize, strides, padsBegin, padsEnd, dilations, paddingType,
                                                      numOutChannels, numGroups);

    auto biasNode = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, std::vector<size_t>{16, 1, 1});
    auto add = std::make_shared<ngraph::opset1::Add>(conv, biasNode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "BinConvFuseConv");
}

void ExecGraphInputsFusingBinConv::TearDown() {
}

TEST_P(ExecGraphInputsFusingBinConv, CheckNumInputsInBinConvFusingWithConv) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::CNNNetwork cnnNet(fnPtr);
    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);

    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();

    if (auto function = execGraphInfo.getFunction()) {
        // try to convert to old representation and check that conversion passed well
        std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedExecGraph;
        ASSERT_NO_THROW(convertedExecGraph = std::make_shared<InferenceEngine::details::CNNNetworkImpl>(execGraphInfo));

        for (const auto & op : function->get_ops()) {
            const auto & rtInfo = op->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                IE_ASSERT(rtInfo.end() != it);
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                IE_ASSERT(nullptr != value);

                return value->get();
            };

            auto layerType = getExecValue("layerType");
            if (layerType == "BinaryConvolution") {
                auto originalLayersNames = getExecValue("originalLayersNames");
                ASSERT_TRUE(originalLayersNames.find("BinaryConvolution") != std::string::npos);
                ASSERT_TRUE(originalLayersNames.find("Add") != std::string::npos);
                ASSERT_EQ(op->get_input_size(), 1);
            }

            // IR v7 does not have output nodes
            if (ngraph::op::is_output(op))
                continue;

            IE_SUPPRESS_DEPRECATED_START
            InferenceEngine::CNNLayerPtr cnnLayer;
            ASSERT_NO_THROW(cnnLayer = CommonTestUtils::getLayerByName(convertedExecGraph.get(), op->get_friendly_name()));
            ASSERT_EQ(cnnLayer->name, op->get_friendly_name());
            auto variantType = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(
                op->get_rt_info()[ExecGraphInfoSerialization::LAYER_TYPE]);;
            ASSERT_EQ(cnnLayer->type, variantType->get());

            for (const auto & kvp : cnnLayer->params) {
                auto variant = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(op->get_rt_info()[kvp.first]);
                ASSERT_EQ(variant->get(), kvp.second);
            }
            IE_SUPPRESS_DEPRECATED_END
        }
    } else {
        IE_SUPPRESS_DEPRECATED_START
        std::vector<InferenceEngine::CNNLayerPtr> nodes;
        ASSERT_NO_THROW(nodes = TopologicalSort(execGraphInfo));
        for (auto &node : nodes) {
            if (node->type == "BinaryConvolution") {
                std::string originalLayersNames = node->params["originalLayersNames"];
                ASSERT_TRUE(originalLayersNames.find("BinaryConvolution") != std::string::npos);
                ASSERT_TRUE(originalLayersNames.find("Add") != std::string::npos);
                ASSERT_EQ(node->insData.size(), 1);
            }
        }
        IE_SUPPRESS_DEPRECATED_END
    }

    fnPtr.reset();
};

}  // namespace ExecutionGraphTests
