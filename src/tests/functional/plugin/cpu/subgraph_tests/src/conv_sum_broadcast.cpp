// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        InputShape, //convShape
        InputShape,  //second term shape
        bool,       // bias flag
        fusingSpecificParams,
        std::map<std::string, std::string> // config
> convSumBroadcastParamSet;


class ConcatConvSumInPlaceTest : public testing::WithParamInterface<convSumBroadcastParamSet>,
                                 virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convSumBroadcastParamSet>& obj) {
        InputShape convShape;
        InputShape secondShape;
        bool bias;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(convShape, secondShape, bias, fusingParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result  << CommonTestUtils::partialShape2str({convShape.first, secondShape.first}) << "_";
        result << "TS=";
        for (const auto& shape : {convShape, secondShape}) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << CommonTestUtils::vec2str(itr);
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

    void SetUp() override {
        InputShape convShape;
        InputShape secondShape;
        bool bias;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::map<std::string, std::string> additionalConfig;
        std::tie(convShape, secondShape, bias, fusingParams, additionalConfig) = this->GetParam();

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        init_input_shapes({convShape, secondShape});

        const InferenceEngine::SizeVector kernel = {3, 3};
        const InferenceEngine::SizeVector stride = {1, 1};
        const InferenceEngine::SizeVector dilation = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const size_t convOutChannels = 64;

        auto netType = ngraph::element::f32;
        auto inputParams = ngraph::builder::makeDynamicParams(netType, inputDynamicShapes);

        auto conv = ngraph::builder::makeConvolution(inputParams[0], ngraph::element::f32, kernel, stride, padBegin,
                                                     padEnd, dilation, ngraph::op::PadType::EXPLICIT, convOutChannels);
        if (bias) {
            auto biasNode = ngraph::builder::makeConstant<float>(ngraph::element::Type_t::f32, ngraph::Shape({1, convOutChannels, 1, 1}), {}, true);
            conv = std::make_shared<ngraph::opset3::Add>(conv, biasNode);
        }

        auto sum = std::make_shared<ngraph::opset3::Add>(conv, inputParams[1]);

        fusedOps.insert(fusedOps.begin(), "Add"); // as we always fuse the sum first

        auto runtimeType = netType;
        if (configuration.count(PluginConfigParams::KEY_ENFORCE_BF16) &&
            PluginConfigParams::YES == configuration[PluginConfigParams::KEY_ENFORCE_BF16].as<std::string>()) {
            runtimeType = ngraph::element::Type_t::bf16;
        }

        selectedType = makeSelectedTypeStr(getPrimitiveType(), runtimeType);

        function = makeNgraphFunction(netType, inputParams, sum, "ConvolutionSumBroadcast");
        targetDevice = CommonTestUtils::DEVICE_CPU;
    }
};

TEST_P(ConcatConvSumInPlaceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {
const auto fusingMulAddFQMullAdd = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(inpNode, constNode);
        }, "Multiply(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
        }, "Add(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto localPrc = inpNode->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(inpNode, constNode);
        }, "Multiply(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params) {
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            auto constNode = ngraph::builder::makeConstant(ngPrc, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(inpNode, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingDivSubFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape = generatePerChannelShape(inpNode);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(inpNode, secondMultInput);
        }, "Divide(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            ngraph::Shape secondMultInShape = generatePerChannelShape(inpNode);
            auto secondMultInput = ngraph::builder::makeConstant(ngPrc, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(inpNode, secondMultInput);
        }, "Subtract(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto localPrc = inpNode->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };

const auto fusingSigmoidFQFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Sigmoid);
        }, "Sigmoid"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto localPrc = inpNode->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto localPrc = inpNode->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"Sigmoid", "FakeQuantize", "FakeQuantize"} };

const auto fusingClampFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            return ngraph::builder::makeActivation(inpNode, ngPrc, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
        }, "Clamp"},
        {[](std::shared_ptr<ngraph::Node> inpNode, const ngraph::element::Type& ngPrc, ngraph::ParameterVector& params){
            auto localPrc = inpNode->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(inpNode);
            return ngraph::builder::makeFakeQuantize(inpNode, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };



const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
        fusingSigmoid,
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        fusingFQPerChannelSigmoidFQPerChannel,
        fusingReluScaleShift,
        fusingMulAddFQMullAdd,
        fusingSigmoidFQFQ,
//        fusingClampFQ // TODO: we need investigation, this particular pattern does not work even in static case
        fusingDivSubFQ
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        fusingSigmoid,
        fusingReluScaleShift
};

InputShape convInpShape = {
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

InputShape secondInp = {
        //dynamic shapes
        {-1, -1, -1, -1},
        { //target static shapes
            {1, 64, 1, 8},
            {1, 64, 1, 8},
            {1, 64, 8, 8},
            {1, 64, 8, 8},
            {1, 64, 8, 1}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP32, ConcatConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::Values(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_BF16, ConcatConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::Values(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
