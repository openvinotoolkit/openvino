// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace ov::test;
using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
using FQConvBiasParams = std::tuple<InputShape, std::string>;

static std::shared_ptr<ov::Model> initFunction(const ov::PartialShape& input_shape, const std::string& layer_type) {
    const auto precision = element::f32;
        auto input_params = builder::makeDynamicParams(precision, {input_shape});
        auto il = opset1::Constant::create(precision, {}, {0.f});
        auto ih = opset1::Constant::create(precision, {}, {12.5f});
        auto ol = opset1::Constant::create(precision, {}, {0.f});
        auto oh = opset1::Constant::create(precision, {}, {12.5f});
        auto fq = std::make_shared<opset1::FakeQuantize>(input_params[0], il, ih, ol, oh, 256);

        std::shared_ptr<Node> layer;
        if (layer_type == "Convolution")  {
            const size_t out_channels = 30;
            const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
            auto weights = builder::makeConstant<int8_t>(ov::element::i8, Shape{out_channels, in_channels, 1, 1}, {}, true);
            auto convert = std::make_shared<ov::opset10::Convert>(weights, precision);
            auto mul_const = builder::makeConstant<float>(precision, Shape{1, 1, 1, 1}, {}, true);
            auto mul = std::make_shared<ov::opset10::Multiply>(convert, mul_const);

            const ov::Strides strides = {1, 1};
            const ov::CoordinateDiff pads_begin = {0, 0};
            const ov::CoordinateDiff pads_end = {0, 0};
            const ov::Strides dilations = {1, 1};
            layer = std::make_shared<ov::opset10::Convolution>(fq, mul, strides, pads_begin, pads_end, dilations);
        } else if (layer_type == "GroupConvolution") {
            const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
            auto weights = builder::makeConstant<int8_t>(ov::element::i8, Shape{in_channels, 1, 1, 1}, {}, true);
            auto convert = std::make_shared<ov::opset10::Convert>(weights, precision);
            auto mul_const = builder::makeConstant<float>(precision, Shape{1, 1, 1, 1}, {}, true);
            auto mul = std::make_shared<ov::opset10::Multiply>(convert, mul_const);
            auto reshape_const = ov::opset10::Constant::create(ov::element::i32, {5}, std::vector<int32_t>{static_cast<int32_t>(in_channels), 1, 1, 1, 1});
            auto reshape = std::make_shared<ov::opset10::Reshape>(mul, reshape_const, true);

            const ov::Strides strides = {1, 1};
            const ov::CoordinateDiff pads_begin = {0, 0};
            const ov::CoordinateDiff pads_end = {0, 0};
            const ov::Strides dilations = {1, 1};
            layer = std::make_shared<ov::opset10::GroupConvolution>(fq, reshape, strides, pads_begin, pads_end, dilations);
        } else if (layer_type == "ConvolutionBackpropData") {
            const size_t out_channels = 30;
            const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
            auto weights = builder::makeConstant<int8_t>(ov::element::i8, Shape{in_channels, out_channels, 3, 3}, {}, true);
            auto convert = std::make_shared<ov::opset10::Convert>(weights, precision);
            auto mul_const = builder::makeConstant<float>(precision, Shape{1, 1, 1, 1}, {}, true);
            auto mul = std::make_shared<ov::opset10::Multiply>(convert, mul_const);

            const ov::Strides strides = {1, 1};
            const ov::CoordinateDiff pads_begin = {0, 0};
            const ov::CoordinateDiff pads_end = {0, 0};
            const ov::Strides dilations = {1, 1};
            layer = std::make_shared<ov::opset10::ConvolutionBackpropData>(fq, mul, strides, pads_begin, pads_end, dilations);
        } else if (layer_type == "MatMul") {
            auto new_param = std::make_shared<ov::opset10::Parameter>(precision, input_shape);
            input_params.push_back(new_param);
            auto il_2 = opset1::Constant::create(precision, {}, {-12.8f});
            auto ih_2 = opset1::Constant::create(precision, {}, {12.7f});
            auto ol_2 = opset1::Constant::create(precision, {}, {-12.8f});
            auto oh_2 = opset1::Constant::create(precision, {}, {12.7f});
            auto fq_2 = std::make_shared<opset1::FakeQuantize>(new_param, il_2, ih_2, ol_2, oh_2, 256);
            layer = std::make_shared<ov::opset10::MatMul>(fq, fq_2, false, true);
        } else if (layer_type == "MatMulWithConstant") {
            const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
            auto weights = builder::makeConstant<int8_t>(ov::element::i8, Shape{in_channels, in_channels}, {}, true);
            auto convert = std::make_shared<ov::opset10::Convert>(weights, precision);
            auto mul_const = builder::makeConstant<float>(precision, Shape{in_channels, 1}, {}, true);
            auto mul = std::make_shared<ov::opset10::Multiply>(convert, mul_const);
            layer = std::make_shared<ov::opset10::MatMul>(fq, mul, false, true);
        } else {
            IE_THROW() << "Unsupported layer type";
        }

        layer->set_friendly_name(layer_type);
        const auto& out_shape = layer->get_output_partial_shape(0);
        Shape bias_shape(out_shape.size(), 1);
        if (layer_type != "MatMul") {
            bias_shape[1] = out_shape[1].get_length();
        }
        auto bias_const = builder::makeConstant<float>(precision, bias_shape, {}, true);
        auto bias = std::make_shared<ov::opset10::Add>(layer, bias_const);
        return std::make_shared<ngraph::Function>(bias, input_params, "FQConvBias");
}

class FQConvBias : virtual public SubgraphBaseTest, public CpuTestWithFusing,
                   public testing::WithParamInterface<FQConvBiasParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FQConvBiasParams> obj) {
        InputShape input_shape;
        std::string layer_type;
        std::tie(input_shape, layer_type) = obj.param;

        std::ostringstream result;
        result << "IS=(" << CommonTestUtils::partialShape2str({input_shape.first}) << ")_TS=(";
        for (const auto& item : input_shape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << ")_layer_type=" << layer_type;
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape input_shape;
        std::string layer_type;
        std::tie(input_shape, layer_type) = GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        fusedOps = std::vector<std::string>{layer_type, "Multiply", "Add"};
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        std::unordered_map<std::string, std::string> ngraph_type_to_plugin_type{
            {"Convolution", "Convolution"},
            {"GroupConvolution", "Convolution"},
            {"ConvolutionBackpropData", "Deconvolution"},
            {"MatMul", "MatMul"},
            {"MatMulWithConstant", "FullyConnected"},
        };
        node_type = ngraph_type_to_plugin_type[layer_type];

        const auto shapes = layer_type == "MatMul" ? std::vector<InputShape>{input_shape, input_shape}
                                                   : std::vector<InputShape>{input_shape};
        init_input_shapes(shapes);
        function = initFunction(inputDynamicShapes[0], layer_type);
    }

    std::string node_type;
};

TEST_P(FQConvBias, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, node_type);
}

namespace {
const std::vector<InputShape> input_shapes_4D = {
    {{}, {{1, 3, 1, 1}}},
    {{-1, 3, -1, -1}, {{1, 3, 256, 256}}}
};

const std::vector<std::string> layer_types_4D = {
    "Convolution",
    "GroupConvolution",
    "ConvolutionBackpropData",
    "MatMul",
};

INSTANTIATE_TEST_SUITE_P(smoke_FQConvBias_4D, FQConvBias,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_4D),
                                            ::testing::ValuesIn(layer_types_4D)),
                         FQConvBias::getTestCaseName);

const std::vector<InputShape> input_shapes_2D = {
    {{-1, 768}, {{1, 768}}}
};

const std::vector<std::string> layer_types_2D = {
    "MatMulWithConstant",
};

INSTANTIATE_TEST_SUITE_P(smoke_FQConvBias_2D, FQConvBias,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_2D),
                                            ::testing::ValuesIn(layer_types_2D)),
                         FQConvBias::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
