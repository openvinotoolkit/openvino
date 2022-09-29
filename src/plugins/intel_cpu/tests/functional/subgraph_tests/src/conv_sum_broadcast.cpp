// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_ops/type_relaxed.hpp>
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

    virtual ngraph::ParameterVector makeParams() {
        return ngraph::builder::makeDynamicParams(ngraph::element::f32, inputDynamicShapes);
    }

    virtual std::shared_ptr<ngraph::Node> makeConv(const ngraph::ParameterVector& inputParams) {
        auto conv = ngraph::builder::makeConvolution(inputParams[0], ngraph::element::f32, _kernel, _stride, _padBegin,
                                                     _padEnd, _dilation, ngraph::op::PadType::EXPLICIT, _convOutChannels);

        return conv;
    }

    virtual std::shared_ptr<ngraph::Node> addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) {
        auto sum = std::make_shared<ngraph::opset3::Add>(lastNode, inputParams[1]);

        fusedOps.insert(fusedOps.begin(), "Add"); // as we always fuse the sum first
        return sum;
    }

    virtual ov::element::Type getNetType() const {
        return ov::element::Type_t::f32;
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

        selectedType = "?";

        function = makeNgraphFunction(getNetType(), inputParams, sum, "ConvolutionSumBroadcast");

        targetDevice = CommonTestUtils::DEVICE_CPU;
    }

protected:
    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(runtimeType == ov::element::Type_t::f32);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == makeSelectedTypeStr(std::string("jit_") + isaType, runtimeType)
                || primType == makeSelectedTypeStr(std::string("brgconv_") + isaType, runtimeType);
    }

protected:
    ov::element::Type runtimeType;
    const InferenceEngine::SizeVector _kernel = {3, 3};
    const InferenceEngine::SizeVector _stride = {1, 1};
    const InferenceEngine::SizeVector _dilation = {1, 1};
    const std::vector<ptrdiff_t> _padBegin = {0, 0};
    const std::vector<ptrdiff_t> _padEnd = {0, 0};
    const size_t _convOutChannels = 64;
};

TEST_P(ConcatConvSumInPlaceTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConcatConvSumInPlaceTestInt8 : public ConcatConvSumInPlaceTest {
public:
    ngraph::ParameterVector makeParams() override {
        ngraph::ParameterVector outs(2);
        outs[0] = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, inputDynamicShapes[0]);
        outs[1] = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputDynamicShapes[1]);
        return outs;
    }

    std::shared_ptr<ngraph::Node> makeConv(const ngraph::ParameterVector& inputParams) override {
        using namespace ngraph;
        auto inputParamsFP32 = builder::makeDynamicParams(element::f32, { inputParams.front()->get_partial_shape() });

        auto convolutionNodeRelaxed = std::make_shared<op::TypeRelaxed<opset1::Convolution>>(
                *as_type_ptr<opset1::Convolution>(builder::makeConvolution(inputParamsFP32.front(), element::f32, _kernel, _stride, _padBegin,
                                                                          _padEnd, _dilation, ngraph::op::PadType::EXPLICIT, _convOutChannels)),
                element::f32);

        auto inpShape = inputParams.front()->get_partial_shape();
        Shape filterShape = {_convOutChannels, static_cast<size_t>(inpShape[1].get_length())};
        filterShape.insert(filterShape.end(), _kernel.begin(), _kernel.end());
        auto filterWeightsNode = builder::makeConstant<int8_t>(element::i8, filterShape, {}, true);

        auto conv = convolutionNodeRelaxed->copy_with_new_inputs({inputParams.front(), filterWeightsNode});

        return conv;
    }

    std::shared_ptr<ngraph::Node> addSum(std::shared_ptr<ngraph::Node> lastNode, const ngraph::ParameterVector& inputParams) override {
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

    void SetUp() override {
        abs_threshold = 1.001f;
        using ngraph::pass::ConvertPrecision;
        ConcatConvSumInPlaceTest::SetUp();
        functionRefs = ov::clone_model(*function);
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::i8, ngraph::element::Type_t::f32>().run_on_function(functionRefs);
        ngraph::pass::ConvertPrecision<ngraph::element::Type_t::u8, ngraph::element::Type_t::f32>().run_on_function(functionRefs);
        functionRefs->validate_nodes_and_infer_types();
    }
};

TEST_P(ConcatConvSumInPlaceTestInt8, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {
const auto fusingMulAddFQMullAdd = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingDivSubFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerChannel)"},
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };

const auto fusingSigmoidFQFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"Sigmoid", "FakeQuantize", "FakeQuantize"} };

const auto fusingClampFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
        }, "Clamp"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
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

const std::vector<InputShape> secondInp = {
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

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP32, ConcatConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_BF16, ConcatConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_INT8, ConcatConvSumInPlaceTestInt8,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConcatConvSumInPlaceTest::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
