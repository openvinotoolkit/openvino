// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_sum_broadcast.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
std::string ConvSumInPlaceTest::getTestCaseName(const testing::TestParamInfo<convSumBroadcastParamSet>& obj) {
    InputShape convShape;
    InputShape secondShape;
    bool bias;
    fusingSpecificParams fusingParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(convShape, secondShape, bias, fusingParams, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "IS=";
    result  << ov::test::utils::partialShape2str({convShape.first, secondShape.first}) << "_";
    result << "TS=";
    for (const auto& shape : {convShape, secondShape}) {
        result << "(";
        if (!shape.second.empty()) {
            for (const auto& itr : shape.second) {
                result << ov::test::utils::vec2str(itr);
            }
        }
        result << ")_";
    }
    result << "bias=" << (bias ? "True" : "False");
    result << CpuTestWithFusing::getTestCaseName(fusingParams);
    if (!additionalConfig.empty()) {
        result << "_PluginConf";
        for (auto& item : additionalConfig) {
            result << "_" << item.first << "=" << item.second;
        }
    }
    return result.str();
}

ngraph::ParameterVector ConvSumInPlaceTest::makeParams() {
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngraph::element::f32, shape));
    }
    return params;
}

std::shared_ptr<ngraph::Node> ConvSumInPlaceTest::makeConv(const ngraph::ParameterVector& inputParams) {
    auto conv = ngraph::builder::makeConvolution(inputParams[0], ngraph::element::f32, _kernel, _stride, _padBegin,
                                                 _padEnd, _dilation, ngraph::op::PadType::EXPLICIT, _convOutChannels);
    return conv;
}

std::shared_ptr<ngraph::Node> ConvSumInPlaceTest::addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) {
    auto sum = std::make_shared<ngraph::opset3::Add>(lastNode, inputParams[1]);
    fusedOps.insert(fusedOps.begin(), "Add"); // as we always fuse the sum first
    return sum;
}

ov::element::Type ConvSumInPlaceTest::getNetType() const {
    return ov::element::Type_t::f32;
}

void ConvSumInPlaceTest::SetUp() {
    InputShape convShape;
    InputShape secondShape;
    bool bias;
    fusingSpecificParams fusingParams;
    std::map<std::string, std::string> additionalConfig;
    std::tie(convShape, secondShape, bias, fusingParams, additionalConfig) = this->GetParam();
    std::tie(postOpMgrPtr, fusedOps) = fusingParams;
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    init_input_shapes({convShape, secondShape});
    auto inputParams = makeParams();
    auto conv = makeConv(inputParams);
    if (bias) {
        auto biasNode = ngraph::builder::makeConstant<float>(ngraph::element::Type_t::f32, ngraph::Shape({1, _convOutChannels, 1, 1}), {}, true);
        conv = std::make_shared<ngraph::opset3::Add>(conv, biasNode);
    }
    auto sum = addSum(conv, inputParams);
    runtimeType = getNetType();
    if (configuration.count(PluginConfigParams::KEY_ENFORCE_BF16) &&
        PluginConfigParams::YES == configuration[PluginConfigParams::KEY_ENFORCE_BF16].as<std::string>()) {
        runtimeType = ngraph::element::Type_t::bf16;
    }
    if (inputParams.front()->get_element_type() == ngraph::element::i8 || inputParams.front()->get_element_type() == ngraph::element::u8) {
        runtimeType = ngraph::element::i8;
    }
    function = makeNgraphFunction(getNetType(), inputParams, sum, "ConvolutionSumBroadcast");
    targetDevice = ov::test::utils::DEVICE_CPU;
    if (!configuration.count(InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE)) {
        configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::DISABLE});
    }
}

TEST_P(ConvSumInPlaceTest, CompareWithRefs) {
    run();
}

ConvSumInPlaceStrided::ConvSumInPlaceStrided() {
    _kernel = {1, 1};
    _stride = {2, 2};
    _convOutChannels = 128;
    rel_threshold = 1e-4;
}

TEST_P(ConvSumInPlaceStrided, CompareWithRefs) {
    run();
}

ngraph::ParameterVector ConvSumInPlaceTestInt8::makeParams() {
    ngraph::ParameterVector outs(2);
    outs[0] = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, inputDynamicShapes[0]);
    outs[1] = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputDynamicShapes[1]);
    return outs;
}

std::shared_ptr<ngraph::Node> ConvSumInPlaceTestInt8::makeConv(const ngraph::ParameterVector& inputParams) {
    using namespace ngraph;
    auto inputParamsFP32 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputParams.front()->get_partial_shape());
    auto convolutionNodeRelaxed = std::make_shared<ov::op::TypeRelaxed<opset1::Convolution>>(
            *as_type_ptr<opset1::Convolution>(builder::makeConvolution(inputParamsFP32, element::f32, _kernel, _stride, _padBegin,
                                                                      _padEnd, _dilation, ngraph::op::PadType::EXPLICIT, _convOutChannels)),
            element::f32);
    auto inpShape = inputParams.front()->get_partial_shape();
    Shape filterShape = {_convOutChannels, static_cast<size_t>(inpShape[1].get_length())};
    filterShape.insert(filterShape.end(), _kernel.begin(), _kernel.end());
    auto filterWeightsNode = builder::makeConstant<int8_t>(element::i8, filterShape, {}, true);
    auto conv = convolutionNodeRelaxed->copy_with_new_inputs({inputParams.front(), filterWeightsNode});
    return conv;
}

std::shared_ptr<ngraph::Node> ConvSumInPlaceTestInt8::addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) {
    std::vector<std::string> additionalFusedOps;
    lastNode = ngraph::builder::makeActivation(lastNode, ngraph::element::f32, ngraph::helpers::Relu);
    //additionalFusedOps.push_back("Relu");
    auto fqShape = ngraph::Shape(lastNode->get_output_partial_shape(0).size(), 1);
    lastNode = ngraph::builder::makeFakeQuantize(lastNode, ngraph::element::f32, 256, fqShape);
    additionalFusedOps.push_back("FakeQuantize");
    auto secondTerm = ngraph::builder::makeFakeQuantize(inputParams[1], ngraph::element::f32, 256, fqShape);
    auto sum = std::make_shared<ngraph::opset3::Add>(lastNode, secondTerm);
    additionalFusedOps.push_back("Add");
    fusedOps.insert(fusedOps.begin(), additionalFusedOps.begin(), additionalFusedOps.end());
    return sum;
}

void ConvSumInPlaceTestInt8::SetUp() {
    abs_threshold = 1.001f;
    using ngraph::pass::ConvertPrecision;
    ConvSumInPlaceTest::SetUp();
    functionRefs = function->clone();
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32>().run_on_model(functionRefs);
    ngraph::pass::ConvertPrecision<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32>().run_on_model(functionRefs);
    functionRefs->validate_nodes_and_infer_types();
}

TEST_P(ConvSumInPlaceTestInt8, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

std::shared_ptr<ngraph::Node> ConvSumInPlaceTestSeveralConsumers::addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) {
    auto sum = std::make_shared<ngraph::opset3::Add>(lastNode, inputParams[1]);
    fusedOps.insert(fusedOps.begin(), "Add");

    auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(sum);
    return std::make_shared<ngraph::opset3::Reshape>(sum, shapeOf, true);
}

TEST_P(ConvSumInPlaceTestSeveralConsumers, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace ConvSumBroadcast {
const InputShape convInpShape() {
    static const InputShape convInpShape = {
            //dynamic shapes
            {-1, 32, -1, -1},
            { //target static shapes
                {1, 32, 10, 10},
                {1, 32, 10, 10},
                {1, 32, 10, 10},
                {1, 32, 3, 3},
                {1, 32, 3, 10}
            }
    };
    return convInpShape;
}

const std::vector<InputShape> secondInp() {
    static const std::vector<InputShape> secondInp = {
        {
            //dynamic shapes
            {-1, -1, -1, -1},
            { //target static shapes
                {1, 64, 1, 8},
                {1, 64, 1, 8},
                {1, 64, 8, 8},
                {1, 64, 8, 8},
                {1, 64, 8, 1}
            }
        },
        {
            {1, 64, 8, 8},
            {
                {1, 64, 8, 8}
            }
        },
    };
    return secondInp;
}

} // namespace ConvSumBroadcast
} // namespace SubgraphTestsDefinitions
