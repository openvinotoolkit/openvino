// Copyright (C) 2018-2023 Intel Corporation
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

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "execution_graph_tests/runtime_precision.hpp"

namespace ExecutionGraphTests {

std::shared_ptr<ngraph::Function> makeEltwiseFunction(const std::vector<InferenceEngine::Precision>& inputPrecisions) {
    IE_ASSERT(inputPrecisions.size() == 2);

    ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[0]),
                                                                       ov::Shape{1, 16, 5, 4}),
                               std::make_shared<ov::op::v0::Parameter>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[1]),
                                                                       ov::Shape{1, 16, 5, 4})};

    auto eltwise = ngraph::builder::makeEltwise(inputs[0], inputs[1], ngraph::helpers::EltwiseTypes::ADD);
    eltwise->set_friendly_name("Eltwise");

    auto function = std::make_shared<ngraph::Function>(eltwise, inputs, "EltwiseWithTwoDynamicInputs");
    return function;
}

std::shared_ptr<ngraph::Function> makeFakeQuantizeReluFunction(const std::vector<InferenceEngine::Precision>& inputPrecisions) {
    IE_ASSERT(inputPrecisions.size() == 1);

    ov::ParameterVector inputs{
        std::make_shared<ov::op::v0::Parameter>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[0]), ov::Shape{1, 16, 5, 4})};
    auto inputLowNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {0});
    auto inputHighNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {255});
    auto outputLowNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {0});
    auto outputHighNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {255});
    auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(inputs[0], inputLowNode, inputHighNode, outputLowNode, outputHighNode, 256);
    fakeQuantize->set_friendly_name("FakeQuantize");

    auto relu = std::make_shared<ngraph::op::Relu>(fakeQuantize);
    relu->set_friendly_name("Relu");

    auto function = std::make_shared<ngraph::Function>(relu, inputs, "FakeQuantizeRelu");
    return function;
}

std::shared_ptr<ngraph::Function> makeFakeQuantizeBinaryConvolutionFunction(const std::vector<InferenceEngine::Precision> &inputPrecisions) {
    IE_ASSERT(inputPrecisions.size() == 1);

    ov::ParameterVector inputs{
        std::make_shared<ov::op::v0::Parameter>(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecisions[0]), ov::Shape{1, 16, 5, 4})};
    auto inputLowNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {1});
    auto inputHighNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {1});
    auto outputLowNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {0});
    auto outputHighNode = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1, 1, 1, 1}, {1});
    auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(inputs[0], inputLowNode, inputHighNode, outputLowNode, outputHighNode, 2);
    fakeQuantize->set_friendly_name("FakeQuantize");

    auto binConv = ngraph::builder::makeBinaryConvolution(fakeQuantize, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, ngraph::op::PadType::EXPLICIT, 32, 0);
    binConv->set_friendly_name("BinaryConvolution");

    auto function = std::make_shared<ngraph::Function>(binConv, inputs, "FakeQuantizeBinaryConvolution");
    return function;
}

std::string ExecGraphRuntimePrecision::getTestCaseName(testing::TestParamInfo<ExecGraphRuntimePrecisionParams> obj) {
    RuntimePrecisionSpecificParams specificParams;
    std::string targetDevice;
    std::tie(specificParams, targetDevice) = obj.param;

    std::ostringstream result;
    result << "Function=" << specificParams.makeFunction(specificParams.inputPrecisions)->get_friendly_name() << "_";
    result << "InPrcs=" << ov::test::utils::vec2str(specificParams.inputPrecisions) << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void ExecGraphRuntimePrecision::SetUp() {
    RuntimePrecisionSpecificParams specificParams;
    std::tie(specificParams, targetDevice) = this->GetParam();
    expectedPrecisions = specificParams.expectedPrecisions;
    fnPtr = specificParams.makeFunction(specificParams.inputPrecisions);
}

void ExecGraphRuntimePrecision::TearDown() {
}

TEST_P(ExecGraphRuntimePrecision, CheckRuntimePrecision) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    InferenceEngine::CNNNetwork cnnNet(fnPtr);
    auto ie = PluginCache::get().ie();
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
    const auto execGraph = execNet.GetExecGraphInfo().getFunction();

    auto ops = execGraph->get_ops();
    for (auto expectedPrc : expectedPrecisions) {
        auto opIter = std::find_if(ops.begin(), ops.end(), [&expectedPrc](std::shared_ptr<ngraph::Node> op) {
            return op->get_friendly_name() == expectedPrc.first;
        });

        if (opIter == ops.end())
            FAIL() << "Execution graph doesn't contain node with name: " << expectedPrc.first;

        const auto& rtInfo = opIter->get()->get_rt_info();
        const auto& rtIter = rtInfo.find("runtimePrecision");

        if (rtIter == rtInfo.end())
            FAIL() << "Runtime precision is not found for node: " << opIter->get()->get_friendly_name();

        if (InferenceEngine::details::convertPrecision(expectedPrc.second).get_type_name() != rtIter->second.as<std::string>())
            FAIL() << "`" << expectedPrc.first << "' node runtime precision mismatch: actual = " <<
                rtIter->second.as<std::string>() << ", expected = " << InferenceEngine::details::convertPrecision(expectedPrc.second).get_type_name();
    }

    fnPtr.reset();
};

}  // namespace ExecutionGraphTests
