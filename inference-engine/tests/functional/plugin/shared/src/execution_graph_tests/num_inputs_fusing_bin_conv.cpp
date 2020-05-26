// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"

#include "network_serializer.h"

namespace LayerTestsDefinitions {

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
    if (targetDevice.find(CommonTestUtils::DEVICE_GPU) != std::string::npos) {
        PluginCache::get().reset();
    }
}

TEST_P(ExecGraphInputsFusingBinConv, CheckNumInputsInBinConvFusingWithConv) {
    InferenceEngine::CNNNetwork cnnNet(fnPtr);
    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);

    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetwork execGraphInfo = execNet.GetExecGraphInfo();
    std::vector<InferenceEngine::CNNLayerPtr> nodes;
    ASSERT_NO_THROW(nodes = InferenceEngine::Serialization::TopologicalSort(execGraphInfo));
    for (auto &node : nodes) {
        if (node->type == "BinaryConvolution") {
            std::string originalLayersNames = node->params["originalLayersNames"];
            ASSERT_TRUE(originalLayersNames.find("BinaryConvolution") != std::string::npos);
            ASSERT_TRUE(originalLayersNames.find("Add") != std::string::npos);
            ASSERT_EQ(node->insData.size(), 1);
        }
    }
    IE_SUPPRESS_DEPRECATED_END

    fnPtr.reset();
};

}  // namespace LayerTestsDefinitions
