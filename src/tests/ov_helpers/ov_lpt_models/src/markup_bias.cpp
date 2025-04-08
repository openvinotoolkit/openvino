// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/markup_bias.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace ov {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> MarkupBiasFunction::get(const ov::element::Type& precision,
                                                   const ov::PartialShape& input_shape,
                                                   const ov::PartialShape& add_shape,
                                                   const std::string& layer_type,
                                                   const bool extra_multipy) {
    ov::ParameterVector input_params{std::make_shared<ov::op::v0::Parameter>(precision, input_shape)};
    auto il = ov::opset1::Constant::create(precision, {}, {0.f});
    auto ih = ov::opset1::Constant::create(precision, {}, {12.5f});
    auto ol = ov::opset1::Constant::create(precision, {}, {0.f});
    auto oh = ov::opset1::Constant::create(precision, {}, {12.5f});
    auto fq = std::make_shared<ov::opset1::FakeQuantize>(input_params[0], il, ih, ol, oh, 256);

    std::shared_ptr<ov::Node> layer;
    const size_t out_channels = 10;
    if (layer_type == "Convolution") {
        const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
        auto weights = ov::test::utils::make_constant(ov::element::i8, Shape{out_channels, in_channels, 1, 1});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, precision);
        auto mul_const = ov::test::utils::make_constant(precision, Shape{1, 1, 1, 1});
        auto mul = std::make_shared<ov::opset1::Multiply>(convert, mul_const);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        layer = std::make_shared<ov::opset1::Convolution>(fq, mul, strides, pads_begin, pads_end, dilations);
    } else if (layer_type == "GroupConvolution") {
        const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
        auto weights = ov::test::utils::make_constant(ov::element::i8, Shape{in_channels, 1, 1, 1}, {});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, precision);
        auto mul_const = ov::test::utils::make_constant(precision, Shape{1, 1, 1, 1});
        auto mul = std::make_shared<ov::opset1::Multiply>(convert, mul_const);

        std::vector<int32_t> target_shape{static_cast<int32_t>(in_channels), 1, 1, 1, 1};
        auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {5}, target_shape);
        auto reshape = std::make_shared<ov::opset1::Reshape>(mul, reshape_const, true);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        layer = std::make_shared<ov::opset1::GroupConvolution>(fq, reshape, strides, pads_begin, pads_end, dilations);
    } else if (layer_type == "ConvolutionBackpropData") {
        const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
        auto weights = ov::test::utils::make_constant(ov::element::i8, Shape{in_channels, out_channels, 1, 1});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, precision);
        auto mul_const = ov::test::utils::make_constant(precision, Shape{1, 1, 1, 1});
        auto mul = std::make_shared<ov::opset1::Multiply>(convert, mul_const);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        layer = std::make_shared<ov::opset1::ConvolutionBackpropData>(fq, mul, strides, pads_begin, pads_end, dilations);
    } else if (layer_type == "MatMul") {
        auto new_param = std::make_shared<ov::opset1::Parameter>(precision, input_shape);
        input_params.push_back(new_param);
        auto il_2 = ov::opset1::Constant::create(precision, {}, {-128.f});
        auto ih_2 = ov::opset1::Constant::create(precision, {}, {127.f});
        auto ol_2 = ov::opset1::Constant::create(precision, {}, {-128.f});
        auto oh_2 = ov::opset1::Constant::create(precision, {}, {127.f});
        auto fq_2 = std::make_shared<ov::opset1::FakeQuantize>(new_param, il_2, ih_2, ol_2, oh_2, 256);
        layer = std::make_shared<ov::opset1::MatMul>(fq, fq_2, false, true);
    } else if (layer_type == "MatMulWithConstant") {
        const size_t in_channels = input_params[0]->get_partial_shape()[1].get_length();
        auto weights = ov::test::utils::make_constant(ov::element::i8, Shape{out_channels, in_channels});
        auto convert = std::make_shared<ov::opset1::Convert>(weights, precision);
        auto mul_const = ov::test::utils::make_constant(precision, Shape{out_channels, 1});
        auto mul = std::make_shared<ov::opset1::Multiply>(convert, mul_const);
        layer = std::make_shared<ov::opset1::MatMul>(fq, mul, false, true);
    } else {
        throw std::runtime_error("Unsupported layer type");
    }

    layer->set_friendly_name(layer_type);

    const auto& out_shape = layer->get_output_partial_shape(0);

    std::shared_ptr<ov::Node> add_input0 = layer;
    if (extra_multipy) {
        Shape mul_shape{};
        if (out_shape.is_static()) {
            mul_shape.resize(out_shape.size(), 1);
            if (layer_type != "MatMul")
                mul_shape[1] = out_shape[1].get_length();
            else
                mul_shape[out_shape.size() - 1] = out_shape[out_shape.size() - 1].get_length();
        } else {
            mul_shape = Shape{1};
        }
        std::shared_ptr<ov::Node> mul;
        auto mul_input = ov::test::utils::make_constant(precision, mul_shape);
        add_input0 = std::make_shared<ov::opset1::Multiply>(layer, mul_input);
    }
    std::shared_ptr<ov::Node> add_input1;
    // empty add_shape means that add_input1 must be generated automatically
    if (add_shape.is_static() && add_shape.size() == 0) {
        Shape bias_shape(out_shape.size(), 1);
        if (layer_type != "MatMul")
            bias_shape[1] = out_shape[1].get_length();
        add_input1 = ov::test::utils::make_constant(precision, bias_shape);
    } else {
        if (add_shape.is_static()) {
            add_input1 = ov::test::utils::make_constant(precision, add_shape.to_shape());
        } else {
            auto new_param = std::make_shared<ov::opset1::Parameter>(precision, input_shape);
            input_params.push_back(new_param);
            add_input1 = new_param;
        }
    }
    auto add = std::make_shared<ov::opset1::Add>(add_input0, add_input1);
    return std::make_shared<ov::Model>(add, input_params);
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
