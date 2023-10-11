// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <ie_core.hpp>
#include <exec_graph_info.hpp>

#include <ngraph/function.hpp>

#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"

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

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, ov::Shape(inputShapes))};
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
    auto function = execGraphInfo.getFunction();
    ASSERT_NE(function, nullptr);

    for (const auto & op : function->get_ops()) {
        const auto & rtInfo = op->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        auto layerType = getExecValue("layerType");
        if (layerType == "BinaryConvolution") {
            auto originalLayersNames = getExecValue("originalLayersNames");
            ASSERT_TRUE(originalLayersNames.find("BinaryConvolution") != std::string::npos);
            ASSERT_EQ(op->get_input_size(), 2);
        }
    }

    fnPtr.reset();
};

}  // namespace ExecutionGraphTests
