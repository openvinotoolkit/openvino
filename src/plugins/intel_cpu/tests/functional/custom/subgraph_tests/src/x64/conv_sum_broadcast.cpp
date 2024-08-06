// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "internal_properties.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/convolution_params.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<InputShape,  // convShape
                   InputShape,  // second term shape
                   bool,        // bias flag
                   fusingSpecificParams,
                   ov::AnyMap  // config
                   >
    convSumBroadcastParamSet;

class ConvSumInPlaceTest : public testing::WithParamInterface<convSumBroadcastParamSet>,
                           virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convSumBroadcastParamSet>& obj) {
        InputShape convShape;
        InputShape secondShape;
        bool bias;
        fusingSpecificParams fusingParams;
        ov::AnyMap additionalConfig;
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
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

    virtual ov::ParameterVector makeParams() {
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        return params;
    }

    virtual std::shared_ptr<ov::Node> makeConv(const ov::ParameterVector& inputParams) {
        auto conv = ov::test::utils::make_convolution(inputParams[0], ov::element::f32, _kernel, _stride, _padBegin,
                                                     _padEnd, _dilation, ov::op::PadType::EXPLICIT, _convOutChannels);

        return conv;
    }

    virtual std::shared_ptr<ov::Node> addSum(std::shared_ptr<ov::Node> lastNode, const ov::ParameterVector& inputParams) {
        auto sum = std::make_shared<ov::op::v1::Add>(lastNode, inputParams[1]);

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
        ov::AnyMap additionalConfig;
        std::tie(convShape, secondShape, bias, fusingParams, additionalConfig) = this->GetParam();

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        init_input_shapes({convShape, secondShape});

        auto inputParams = makeParams();

        auto conv = makeConv(inputParams);

        if (bias) {
            auto biasNode = ov::test::utils::make_constant(ov::element::Type_t::f32, ov::Shape({1, _convOutChannels, 1, 1}));
            conv = std::make_shared<ov::op::v1::Add>(conv, biasNode);
        }

        auto sum = addSum(conv, inputParams);

        runtimeType = getNetType();
        auto it = configuration.find(ov::hint::inference_precision.name());
        ov::element::Type inference_precision = (it != configuration.end()) ?
                                                it->second.as<ov::element::Type>() : ov::element::undefined;
        if (inference_precision == ov::element::bf16) {
            runtimeType = ov::element::Type_t::bf16;
        } else if (inference_precision == ov::element::f16) {
            runtimeType = ov::element::Type_t::f16;
        }

        if (inputParams.front()->get_element_type() == ov::element::i8 || inputParams.front()->get_element_type() == ov::element::u8) {
            runtimeType = ov::element::i8;
        }

        selectedType = "?";

        function = makeNgraphFunction(getNetType(), inputParams, sum, "ConvolutionSumBroadcast");

        targetDevice = ov::test::utils::DEVICE_CPU;
        if (!configuration.count(ov::intel_cpu::snippets_mode.name())) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }
    }

protected:
    bool primTypeCheck(std::string primType) const override {
        bool skip_amx = false;
        switch (runtimeType) {
            case ov::element::Type_t::f32:
                skip_amx = true;
                break;
            case ov::element::Type_t::f16:
                skip_amx = ov::with_cpu_x86_avx512_core_amx_fp16() ? false : true;
                break;
            case ov::element::Type_t::bf16:
                skip_amx = false;
                break;
            default:
                skip_amx = false;
        }
        auto isaType = getISA(skip_amx);
        const std::regex jit_case_regex(makeSelectedTypeStr(std::string("jit_") + isaType, runtimeType),
                                        std::regex_constants::icase);
        const std::regex brgconv_case_regex(makeSelectedTypeStr(std::string("brgconv_") + isaType, runtimeType),
                                            std::regex_constants::icase);
        if (isaType == "")
            return primType == "ref";
        else
            return std::regex_match(primType, jit_case_regex) || std::regex_match(primType, brgconv_case_regex);
    }

protected:
    ov::element::Type runtimeType;
    ov::Shape _kernel = {3, 3};
    ov::Shape _stride = {1, 1};
    ov::Shape _dilation = {1, 1};
    std::vector<ptrdiff_t> _padBegin = {0, 0};
    std::vector<ptrdiff_t> _padEnd = {0, 0};
    size_t _convOutChannels = 64;
};

TEST_P(ConvSumInPlaceTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvSumInPlaceStrided : public ConvSumInPlaceTest {
public:
    ConvSumInPlaceStrided() {
        _kernel = {1, 1};
        _stride = {2, 2};
        _convOutChannels = 128;
        rel_threshold = 1e-4;
    }

protected:
    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(runtimeType == ov::element::Type_t::f32);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == makeSelectedTypeStr(std::string("jit_") + isaType + std::string("_1x1"), runtimeType)
                || primType == makeSelectedTypeStr(std::string("brgconv_") + isaType+ std::string("_1x1"), runtimeType);
    }
};

TEST_P(ConvSumInPlaceStrided, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvSumInPlaceTestInt8 : public ConvSumInPlaceTest {
public:
    ov::ParameterVector makeParams() override {
        ov::ParameterVector outs(2);
        outs[0] = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes[0]);
        outs[1] = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);
        return outs;
    }

    std::shared_ptr<ov::Node> makeConv(const ov::ParameterVector& inputParams) override {
        auto inputParamsFP32 = std::make_shared<ov::op::v0::Parameter>(element::f32, inputParams.front()->get_partial_shape());

        auto convolutionNodeRelaxed = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Convolution>>(
            *as_type_ptr<ov::op::v1::Convolution>(ov::test::utils::make_convolution(inputParamsFP32,
                                                                                    element::f32,
                                                                                    _kernel,
                                                                                    _stride,
                                                                                    _padBegin,
                                                                                    _padEnd,
                                                                                    _dilation,
                                                                                    ov::op::PadType::EXPLICIT,
                                                                                    _convOutChannels)),
            ov::element::f32);

        auto inpShape = inputParams.front()->get_partial_shape();
        Shape filterShape = {_convOutChannels, static_cast<size_t>(inpShape[1].get_length())};
        filterShape.insert(filterShape.end(), _kernel.begin(), _kernel.end());
        auto filterWeightsNode = ov::test::utils::make_constant(ov::element::i8, filterShape);

        auto conv = convolutionNodeRelaxed->copy_with_new_inputs({inputParams.front(), filterWeightsNode});

        return conv;
    }

    std::shared_ptr<ov::Node> addSum(std::shared_ptr<ov::Node> lastNode, const ov::ParameterVector& inputParams) override {
        std::vector<std::string> additionalFusedOps;

        lastNode = ov::test::utils::make_activation(lastNode, ov::element::f32, ov::test::utils::Relu);
        //additionalFusedOps.push_back("Relu");

        auto fqShape = ov::Shape(lastNode->get_output_partial_shape(0).size(), 1);
        lastNode = ov::test::utils::make_fake_quantize(lastNode, ov::element::f32, 256, fqShape);
        additionalFusedOps.push_back("FakeQuantize");

        auto secondTerm = ov::test::utils::make_fake_quantize(inputParams[1], ov::element::f32, 256, fqShape);

        auto sum = std::make_shared<ov::op::v1::Add>(lastNode, secondTerm);
        additionalFusedOps.push_back("Add");

        fusedOps.insert(fusedOps.begin(), additionalFusedOps.begin(), additionalFusedOps.end());
        return sum;
    }

    void SetUp() override {
        abs_threshold = 1.001f;
        ConvSumInPlaceTest::SetUp();
        functionRefs = function->clone();
        convert_precisions.insert({ov::element::i8, ov::element::f32});
        convert_precisions.insert({ov::element::u8, ov::element::f32});
    }
};

TEST_P(ConvSumInPlaceTestInt8, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvSumInPlaceTestSeveralConsumers : public ConvSumInPlaceTest {
public:
    std::shared_ptr<ov::Node> addSum(std::shared_ptr<ov::Node> lastNode, const ov::ParameterVector& inputParams) override {
        auto sum = std::make_shared<ov::op::v1::Add>(lastNode, inputParams[1]);
        fusedOps.insert(fusedOps.begin(), "Add");

        auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(sum);
        return std::make_shared<ov::op::v1::Reshape>(sum, shapeOf, true);
    }
};

TEST_P(ConvSumInPlaceTestSeveralConsumers, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvSumBroadcastTest : public ConvSumInPlaceTest {
public:
    ConvSumBroadcastTest() {
        _convOutChannels = 8;
        rel_threshold = 1e-4;
    }

protected:
};

TEST_P(ConvSumBroadcastTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class Conv1x1SumUnsupportedBroadcastTest : public ConvSumInPlaceTest {
public:
    Conv1x1SumUnsupportedBroadcastTest() {
        _convOutChannels = 8;
        _kernel = {1, 1};
    }

    std::shared_ptr<ov::Node> addSum(std::shared_ptr<ov::Node> lastNode, const ov::ParameterVector& inputParams) override {
        return std::make_shared<ov::op::v1::Add>(lastNode, inputParams[1]);
    }

protected:
    bool primTypeCheck(std::string primType) const override {
        auto isaType = getISA(runtimeType == ov::element::Type_t::f32);
        if (isaType == "")
            return primType == "ref";
        else
            return primType == makeSelectedTypeStr(std::string("jit_") + isaType + std::string("_1x1"), runtimeType)
                || primType == makeSelectedTypeStr(std::string("brgconv_") + isaType+ std::string("_1x1"), runtimeType);
    }
};

TEST_P(Conv1x1SumUnsupportedBroadcastTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();

    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {
const auto fusingMulAddFQMullAdd = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
            return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
            return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
            return std::make_shared<ov::op::v1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ov::test::utils::make_constant(cfg.type, newShape);
            return std::make_shared<ov::op::v1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingDivSubFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerChannel)"},
        {[](postNodeConfig& cfg){
            ov::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ov::test::utils::make_constant(cfg.type, secondMultInShape);
            return std::make_shared<ov::op::v1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };

const auto fusingSigmoidFQFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ov::test::utils::make_activation(cfg.input, cfg.type, ov::test::utils::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"Sigmoid", "FakeQuantize", "FakeQuantize"} };

const auto fusingClampFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ov::test::utils::make_activation(cfg.input, cfg.type, ov::test::utils::Clamp, {}, {3.0f, 6.0f});
        }, "Clamp"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ov::Shape newShape = generatePerChannelShape(cfg.input);
            return ov::test::utils::make_fake_quantize(cfg.input, localPrc, 256, newShape);
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

const std::vector<fusingSpecificParams> fusingParamsSetFP16 = fusingParamsSetBF16;

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

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP32, ConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_BF16,
                         ConvSumInPlaceTest,
                         ::testing::Combine(::testing::Values(convInpShape),
                                            ::testing::ValuesIn(secondInp),
                                            ::testing::Values(true, false),
                                            ::testing::ValuesIn(fusingParamsSetBF16),
                                            ::testing::Values(cpu_bf16_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP16,
                         ConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSetFP16),
                                 ::testing::Values(cpu_f16_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_INT8, ConvSumInPlaceTestInt8,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_Several_Consumers, ConvSumInPlaceTestSeveralConsumers,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape),
                                 ::testing::ValuesIn(secondInp),
                                 ::testing::Values(true),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

InputShape convInpShapeStrided = {
        //dynamic shapes
        {-1, 64, -1, -1},
        { //target static shapes
            {1, 64, 147, 147},
            {1, 64, 147, 147},
        }
};

InputShape secondInpStrided = {
        //dynamic shapes
        {-1, 128, -1, -1},
        { //target static shapes
            {1, 128, 74, 74},
            {1, 128, 74, 1}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_Strided, ConvSumInPlaceStrided,
                         ::testing::Combine(
                                 ::testing::Values(convInpShapeStrided),
                                 ::testing::Values(secondInpStrided),
                                 ::testing::Values(true),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

InputShape convInpShapeStaticShape = {
        //dynamic shapes
        {-1, 8, {6, 12}, {6, 12}},
        { //target static shapes
            {1, 8, 6, 6}
        }
};

InputShape secondInpStaticShape = {
        //dynamic shapes
        {-1, 8, 1, 1},
        { //target static shapes
            {1, 8, 1, 1}
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_StaticShape, ConvSumBroadcastTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShapeStaticShape),
                                 ::testing::Values(secondInpStaticShape),
                                 ::testing::Values(true),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

InputShape convSum1x1BcastShape1 = {
    // dynamic shapes
    {-1, 8, 1, 1},
    { // target static shapes
        {1, 8, 1, 1}
    }
};

InputShape convSum1x1BcastShape2 = {
    // dynamic shapes
    {-1, 8, -1, -1},
    { // target static shapes
        {1, 8, 6, 6}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_1x1_Broadcast, Conv1x1SumUnsupportedBroadcastTest,
                         ::testing::Combine(
                                 ::testing::Values(convSum1x1BcastShape1),
                                 ::testing::Values(convSum1x1BcastShape2),
                                 ::testing::Values(false),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvSumInPlaceTest::getTestCaseName);

} // namespace
}  // namespace test
}  // namespace ov