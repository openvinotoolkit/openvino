// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity_functions.hpp"

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>

#include <ngraph_transformations/rt_info/mixed_affinity_props.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/builders.hpp"
#include <ov_ops/type_relaxed.hpp>

namespace ov {
namespace test {
namespace mixed_affinity {
namespace {
std::shared_ptr<ov::Node> getDefaultConv(const ov::Output<ov::Node> data, const size_t out_channels) {
    const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    return ngraph::builder::makeConvolution(data, ov::element::f32, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, pad_type, out_channels);
}

std::shared_ptr<ov::Node> getRandomConst(const ov::element::Type& precision, const ov::Shape shape) {
    return ngraph::builder::makeConstant(precision, shape, std::vector<size_t>{}, true);
}
} // namespace

void MixedAffinityFunctionBase::markup_model(const std::shared_ptr<ov::Model>& m, const MixedAffinityMarkup& markup) {
    for (const auto& node : m->get_ordered_ops()) {
        auto it = markup.find(node->get_friendly_name());
        if (it != markup.end()) {
            ov::intel_cpu::mixed_affinity::Properties props(it->second.first, it->second.second);
            ov::intel_cpu::mixed_affinity::set_properties(node, props);
        }
    }
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getOriginal(const MixedAffinityMarkup& markup) {
    const auto original = initOriginal();
    markup_model(original, markup);
    return original;
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getReference() {
    if (std::all_of(input_shapes.begin(), input_shapes.end(), [](const ov::PartialShape& s) { return s[0].get_length() == 0; }))
        return initOriginal();
    else
        return initReference();
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::initOriginal() {
    throw std::runtime_error("initOriginal() is not implemented");
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::initReference() {
    throw std::runtime_error("initReference() is not implemented");
}

ConvWithBiasFunction::ConvWithBiasFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithBiasFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const auto convolution = getDefaultConv(parameters[0], out_channels);
    convolution->set_friendly_name("convolution");

    const auto bias_const = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    const auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
    bias->set_friendly_name("bias");

    return std::make_shared<ov::Model>(ov::NodeVector{bias}, parameters);
}

std::shared_ptr<ov::Model> ConvWithBiasFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    const size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, batch_size);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto bias_const = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});

    const std::string bias_name = "bias";
    const std::string conv_name = "convolution";
    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                           weights_const,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);

        const auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
        bias->set_friendly_name(bias_name + name_suffix);
        concat_inputs[i] = bias;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(bias_name);

    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvAndGrConvFunction::ConvAndGrConvFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvAndGrConvFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 10;
    const size_t in_channels = parameters[0]->get_partial_shape()[1].get_length();
    const auto conv_weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 9, 9});
    const auto convolution = std::make_shared<ov::opset1::Convolution>(parameters[0],
                                                                       conv_weights_const,
                                                                       ov::Strides{2, 2},
                                                                       ov::CoordinateDiff{0, 0},
                                                                       ov::CoordinateDiff{0, 0},
                                                                       ov::Strides{1, 1});
    convolution->set_friendly_name("convolution");

    const auto gr_conv_weights_const = getRandomConst(ov::element::f32, {out_channels, 1, 1, 3, 3});
    auto group_conv = std::make_shared<ov::opset1::GroupConvolution>(convolution,
                                                                     gr_conv_weights_const,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});
    group_conv->set_friendly_name("group_conv");

    return std::make_shared<ov::Model>(ov::NodeVector{group_conv}, parameters);
}

Int8ConvWithDqSubFunction::Int8ConvWithDqSubFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> Int8ConvWithDqSubFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::u8, input_shapes);

    const size_t in_channels = parameters[0]->get_partial_shape()[1].get_length();
    const auto sub_const = ov::opset1::Constant::create(ov::element::u8, {1, in_channels, 1, 1}, {128});
    const auto sub = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
        ov::element::TypeVector{ov::element::f32, ov::element::f32},
        ov::element::TypeVector{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(parameters[0], ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(sub_const, ov::element::f32).get());
    sub->set_friendly_name("sub");

    const size_t out_channels = 30;
    const auto weights_const = getRandomConst(ov::element::i8, {out_channels, in_channels, 3, 3});
    const auto convolution = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
        ov::element::TypeVector{ov::element::f32, ov::element::f32},
        ov::element::TypeVector{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(sub, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
        ov::Strides{1, 1},
        ov::CoordinateDiff{0, 0},
        ov::CoordinateDiff{0, 0},
        ov::Strides{1, 1});
    convolution->set_friendly_name("convolution");
    return std::make_shared<ov::Model>(ov::NodeVector{convolution}, parameters);
}

std::shared_ptr<ov::Model> Int8ConvWithDqSubFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::u8, input_shapes);
    const size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, batch_size);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto sub_const = ov::opset1::Constant::create(ov::element::u8, {1, in_channels, 1, 1}, {128});
    const auto weights_const = getRandomConst(ov::element::i8, {out_channels, in_channels, 3, 3});

    const std::string sub_name = "sub";
    const std::string conv_name = "convolution";
    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto sub = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Subtract>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(split->output(i), ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(sub_const, ov::element::f32).get());
        sub->set_friendly_name(sub_name + name_suffix);
        const auto convolution = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Convolution>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(sub, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
            ov::Strides{1, 1},
            ov::CoordinateDiff{0, 0},
            ov::CoordinateDiff{0, 0},
            ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);
        concat_inputs[i] = convolution;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(conv_name);

    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

GrConvWithParamFunction::GrConvWithParamFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> GrConvWithParamFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    auto group_conv = std::make_shared<ov::opset1::GroupConvolution>(parameters[0],
                                                                     parameters[1],
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});
    group_conv->set_friendly_name("group_conv");
    return std::make_shared<ov::Model>(ov::NodeVector{group_conv}, parameters);
}

std::shared_ptr<ov::Model> GrConvWithParamFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, batch_size);

    const std::string group_conv_name = "group_conv";
    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        auto group_conv = std::make_shared<ov::opset1::GroupConvolution>(split->output(i),
                                                                         parameters[1],
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
        group_conv->set_friendly_name(group_conv_name);
        concat_inputs[i] = group_conv;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(group_conv_name);
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvAndGatherWithParamFunction::ConvAndGatherWithParamFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvAndGatherWithParamFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::i32}, input_shapes);
    const size_t out_channels = 30;
    const auto convolution = getDefaultConv(parameters[0], out_channels);
    convolution->set_friendly_name("convolution");

    auto gather_axis = ov::opset1::Constant::create(ov::element::i32, {}, {1});
    auto gather = std::make_shared<ov::opset10::Gather>(convolution, parameters[1], gather_axis);
    gather->set_friendly_name("gather");

    return std::make_shared<ov::Model>(ov::NodeVector{gather}, parameters);
}

ConvWithTransposeFunction::ConvWithTransposeFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithTransposeFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const auto convolution = getDefaultConv(parameters[0], out_channels);
    convolution->set_friendly_name("convolution");

    const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
    transpose->set_friendly_name("transpose");

    return std::make_shared<ov::Model>(ov::NodeVector{transpose}, parameters);
}

std::shared_ptr<ov::Model> ConvWithTransposeFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    const size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, batch_size);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
    const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});

    const std::string conv_name = "convolution";
    const std::string transpose_name = "transpose";
    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                           weights_const,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);

        const auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
        transpose->set_friendly_name(transpose_name + name_suffix);
        concat_inputs[i] = transpose;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 1);
    concat->set_friendly_name(transpose_name);

    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvWithReshapeFunction::ConvWithReshapeFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithReshapeFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    auto relu = std::make_shared<ov::opset1::Relu>(parameters[0]);
    auto weights_const = getRandomConst(ov::element::f32, {30, 3, 3, 3});
    auto convolution = std::make_shared<ov::opset1::Convolution>(relu,
                                                                 weights_const,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    convolution->set_friendly_name("convolution");

    auto shape_of = std::make_shared<ov::opset10::ShapeOf>(relu, ov::element::i32);
    auto gather_axis = ov::opset1::Constant::create(ov::element::i32, {1}, {0});
    auto gather_indices = ov::opset1::Constant::create(ov::element::i32, {1}, {1});
    auto gather = std::make_shared<ov::opset10::Gather>(shape_of, gather_indices, gather_axis);

    auto zero_const = ov::opset1::Constant::create(ov::element::i32, {1}, {0});
    auto minus_one_const = ov::opset1::Constant::create(ov::element::i32, {1}, {-1});
    auto requested_shape = std::make_shared<ov::opset1::Concat>(ov::OutputVector{zero_const, gather, minus_one_const}, 0);

    auto reshape = std::make_shared<ov::opset1::Reshape>(convolution, requested_shape, true);
    reshape->set_friendly_name("reshape");

    return std::make_shared<ov::Model>(ov::NodeVector{reshape}, parameters);
}

std::shared_ptr<ov::Model> ConvWithReshapeFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    auto relu = std::make_shared<ov::opset1::Relu>(parameters[0]);
    const size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(relu, split_axis, batch_size);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].is_static() ? input_shapes[0][1].get_length() : 3;
    const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});

    const std::string conv_name = "convolution";
    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                           weights_const,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);
        concat_inputs[i] = convolution;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(conv_name);

    auto shape_of = std::make_shared<ov::opset10::ShapeOf>(relu, ov::element::i32);
    auto gather_axis = ov::opset1::Constant::create(ov::element::i32, {1}, {0});
    auto gather_indices = ov::opset1::Constant::create(ov::element::i32, {1}, {1});
    auto gather = std::make_shared<ov::opset10::Gather>(shape_of, gather_indices, gather_axis);

    auto zero_const = ov::opset1::Constant::create(ov::element::i32, {1}, {0});
    auto minus_one_const = ov::opset1::Constant::create(ov::element::i32, {1}, {-1});
    auto requested_shape = std::make_shared<ov::opset1::Concat>(ov::OutputVector{zero_const, gather, minus_one_const}, 0);
    const auto reshape = std::make_shared<ov::opset1::Reshape>(concat, requested_shape, true);
    reshape->set_friendly_name("reshape");

    return std::make_shared<ov::Model>(ov::NodeVector{reshape}, parameters);
}

ConvWithLRNFunction::ConvWithLRNFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithLRNFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const auto convolution = getDefaultConv(parameters[0], out_channels);
    convolution->set_friendly_name("convolution");

    const auto lrn = std::make_shared<ov::opset1::LRN>(convolution, 9e-5, 0.75, 1, 5);
    lrn->set_friendly_name("lrn");

    return std::make_shared<ov::Model>(ov::NodeVector{lrn}, parameters);
}

ConvWithSplitAndResultFunction::ConvWithSplitAndResultFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithSplitAndResultFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 10;
    const auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    auto relu_1 = std::make_shared<ov::opset1::Relu>(convolution_1);
    relu_1->set_friendly_name("relu_1");

    auto split_const = ov::opset1::Constant::create(ov::element::i32, {}, {1});
    auto split = std::make_shared<ov::opset1::Split>(relu_1, split_const, 2);
    split->set_friendly_name("split");

    const auto convolution_2 = getDefaultConv(split->output(1), out_channels);
    convolution_2->set_friendly_name("convolution_2");
    auto relu_2 = std::make_shared<ov::opset1::Relu>(convolution_2);
    relu_2->set_friendly_name("relu_2");

    auto res_1 = std::make_shared<ov::opset1::Result>(split->output(0));
    auto res_2 = std::make_shared<ov::opset1::Result>(relu_2);
    res_1->set_friendly_name("res_1");
    res_2->set_friendly_name("res_2");

    return std::make_shared<ov::Model>(ov::ResultVector{res_1, res_2}, parameters);
}

std::shared_ptr<ov::Model> ConvWithSplitAndResultFunction::initReference() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    size_t batch_size = input_shapes[0][0].get_length();
    const auto start_split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto start_split = std::make_shared<ov::opset1::Split>(parameters[0], start_split_axis, batch_size);

    const size_t out_channels = 10;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const_1 = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
    const auto weights_const_2 = getRandomConst(ov::element::f32, {out_channels, out_channels / 2, 3, 3});
    const auto split_const = ov::opset1::Constant::create(ov::element::i32, {}, {1});

    const std::string conv_name_1 = "convolution_1";
    const std::string conv_name_2 = "convolution_2";
    const std::string relu_name_1 = "relu_1";
    const std::string relu_name_2 = "relu_2";
    const std::string split_name = "split";

    ov::OutputVector concat_inputs_1(batch_size);
    ov::OutputVector concat_inputs_2(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution_1 = std::make_shared<ov::opset1::Convolution>(start_split->output(i),
                                                                             weights_const_1,
                                                                             ov::Strides{1, 1},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::Strides{1, 1});
        const auto relu_1 = std::make_shared<ov::opset1::Relu>(convolution_1);
        convolution_1->set_friendly_name(conv_name_1 + name_suffix);
        auto split = std::make_shared<ov::opset1::Split>(relu_1, split_const, 2);
        split->set_friendly_name(split_name + name_suffix);
        concat_inputs_1[i] = split->output(0);

        const auto convolution_2 = std::make_shared<ov::opset1::Convolution>(split->output(1),
                                                                             weights_const_2,
                                                                             ov::Strides{1, 1},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::Strides{1, 1});
        convolution_2->set_friendly_name(conv_name_2 + name_suffix);
        const auto relu_2 = std::make_shared<ov::opset1::Relu>(convolution_2);
        concat_inputs_2[i] = relu_2;
    }

    const auto concat_1 = ngraph::builder::makeConcat(concat_inputs_1, 0);
    concat_1->set_friendly_name(split_name + ".0");
    const auto concat_2 = ngraph::builder::makeConcat(concat_inputs_2, 0);
    concat_2->set_friendly_name(relu_name_2);

    auto res_1 = std::make_shared<ov::opset1::Result>(concat_1);
    auto res_2 = std::make_shared<ov::opset1::Result>(concat_2);
    res_1->set_friendly_name("res_1");
    res_2->set_friendly_name("res_2");

    return std::make_shared<ov::Model>(ov::ResultVector{res_1, res_2}, parameters);
}

ConvolutionsAndSplitFunction::ConvolutionsAndSplitFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvolutionsAndSplitFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    auto split_const = ov::opset1::Constant::create(ov::element::i32, {}, {1});
    auto split = std::make_shared<ov::opset1::Split>(convolution_1, split_const, 2);
    split->set_friendly_name("split");

    auto s2b_1 = ngraph::builder::makeSpaceToBatch(split->output(0), ov::element::i32, {1, 1, 4, 4}, {0, 0, 0, 0}, {0, 0, 0, 0});
    auto s2b_2 = ngraph::builder::makeSpaceToBatch(split->output(1), ov::element::i32, {1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0});
    s2b_1->set_friendly_name("s2b_1");
    s2b_2->set_friendly_name("s2b_2");

    return std::make_shared<ov::Model>(ov::NodeVector{s2b_1, s2b_2}, parameters);
}

std::shared_ptr<ov::Model> ConvolutionsAndSplitFunction::initReference() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis_1 = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split_1 = std::make_shared<ov::opset1::Split>(parameters[0], split_axis_1, batch_size);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const_1 = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
    const auto split_const = ov::opset1::Constant::create(ov::element::i32, {}, {1});
    const std::string conv_name = "convolution_1";
    const std::string split_name = "split";

    ov::OutputVector concat_inputs_1(batch_size);
    ov::OutputVector concat_inputs_2(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split_1->output(i),
                                                                           weights_const_1,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);
        auto split = std::make_shared<ov::opset1::Split>(convolution, split_const, 2);
        split->set_friendly_name(split_name + name_suffix);
        concat_inputs_1[i] = split->output(0);
        concat_inputs_2[i] = split->output(1);
    }

    const auto concat_1 = ngraph::builder::makeConcat(concat_inputs_1, 0);
    concat_1->set_friendly_name(split_name + ".0");
    const auto concat_2 = ngraph::builder::makeConcat(concat_inputs_2, 0);
    concat_2->set_friendly_name(split_name + ".1");

    auto s2b_1 = ngraph::builder::makeSpaceToBatch(concat_1, ov::element::i32, {1, 1, 4, 4}, {0, 0, 0, 0}, {0, 0, 0, 0});
    auto s2b_2 = ngraph::builder::makeSpaceToBatch(concat_2, ov::element::i32, {1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0});
    s2b_1->set_friendly_name("s2b_1");
    s2b_2->set_friendly_name("s2b_2");

    return std::make_shared<ov::Model>(ov::NodeVector{s2b_1, s2b_2}, parameters);
}

TwoConvWithS2BFunction::TwoConvWithS2BFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> TwoConvWithS2BFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 10;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");
    auto s2b = ngraph::builder::makeSpaceToBatch(convolution_1, ov::element::i32, {1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0});
    s2b->set_friendly_name("s2b");
    auto convolution_2 = getDefaultConv(s2b, out_channels);
    convolution_2->set_friendly_name("convolution_2");

    return std::make_shared<ov::Model>(ov::NodeVector{convolution_2}, parameters);
}

std::shared_ptr<ov::Model> TwoConvWithS2BFunction::initReference() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    size_t batch_size = input_shapes[0][0].get_length();
    const auto split_axis_1 = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split_1 = std::make_shared<ov::opset1::Split>(parameters[0], split_axis_1, batch_size);

    const size_t out_channels = 10;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const_1 = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
    const std::string conv_name_1 = "convolution_1";

    ov::OutputVector concat_inputs_1(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split_1->output(i),
                                                                           weights_const_1,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name_1 + name_suffix);
        concat_inputs_1[i] = convolution;
    }

    const auto concat_1 = ngraph::builder::makeConcat(concat_inputs_1, 0);
    concat_1->set_friendly_name(conv_name_1);
    const size_t block_size = 2;
    auto s2b = ngraph::builder::makeSpaceToBatch(concat_1, ov::element::i32, {1, 1, block_size, block_size}, {0, 0, 0, 0}, {0, 0, 0, 0});
    s2b->set_friendly_name("s2b");

    const size_t new_batch = batch_size * block_size * block_size;
    const auto split_axis_2 = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split_2 = std::make_shared<ov::opset1::Split>(s2b, split_axis_2, new_batch);

    const auto weights_const_2 = getRandomConst(ov::element::f32, {out_channels, out_channels, 3, 3});
    const std::string conv_name_2 = "convolution_2";

    ov::OutputVector concat_inputs_2(new_batch);
    for (size_t i = 0; i < new_batch; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split_2->output(i),
                                                                           weights_const_2,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name_2 + name_suffix);
        concat_inputs_2[i] = convolution;
    }

    const auto concat_2 = ngraph::builder::makeConcat(concat_inputs_2, 0);
    concat_2->set_friendly_name(conv_name_2);

    return std::make_shared<ov::Model>(ov::NodeVector{concat_2}, parameters);
}

TwoConvAndAddFunction::TwoConvAndAddFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> TwoConvAndAddFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    auto bias_const_1 = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    bias_const_1->set_friendly_name("bias_const_1");
    auto bias_1 = ngraph::builder::makeEltwise(convolution_1, bias_const_1, ngraph::helpers::ADD);
    bias_1->set_friendly_name("bias_1");

    auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    auto bias_const_2 = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    bias_const_2->set_friendly_name("bias_const_2");
    auto bias_2 = ngraph::builder::makeEltwise(convolution_2, bias_const_2, ngraph::helpers::ADD);
    bias_2->set_friendly_name("bias_2");

    auto add = ngraph::builder::makeEltwise(bias_1, bias_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

std::shared_ptr<ov::Model> TwoConvAndAddFunction::initReference() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    const size_t bs_left = input_shapes[0][0].get_length();
    const size_t bs_right = input_shapes[1][0].get_length();
    const size_t out_batch = std::max(bs_left, bs_right);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    std::string conv_name = "convolution";
    std::string bias_name = "bias";

    auto get_subgraph = [&](const size_t branch_idx, std::shared_ptr<ov::opset1::Parameter>& param) {
        const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
        const auto bias_const = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
        std::string conv_branch_name = conv_name + "_" + std::to_string(branch_idx);
        std::string bias_branch_name = bias_name + "_" + std::to_string(branch_idx);

        const size_t batch_size = param->get_partial_shape()[0].get_length();
        if (batch_size == 1) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(param,
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name);
            auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
            bias->set_friendly_name(bias_branch_name);
            return ov::OutputVector{bias};
        }

        const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
        const auto split = std::make_shared<ov::opset1::Split>(param, split_axis, batch_size);
        ov::OutputVector res;
        for (size_t i = 0; i < batch_size; ++i) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name + "_" + std::to_string(i));
            auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
            bias->set_friendly_name(bias_branch_name + "_" + std::to_string(i));
            res.push_back(bias);
        }
        return res;
    };

    const auto left_subgraph = get_subgraph(1, parameters[0]);
    const auto right_subgraph = get_subgraph(2, parameters[1]);

    std::string add_name = "add";
    ov::OutputVector concat_inputs(out_batch);
    for (size_t i = 0; i < out_batch; ++i) {
        auto add = ngraph::builder::makeEltwise(left_subgraph.size() == 1 ? left_subgraph[0] : left_subgraph[i],
                                                right_subgraph.size() == 1 ? right_subgraph[0] : right_subgraph[i],
                                                ngraph::helpers::ADD);
        add->set_friendly_name(add_name + "_" + std::to_string(i));
        concat_inputs[i] = add;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(add_name);
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvAndAddWithParameterFunction::ConvAndAddWithParameterFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvAndAddWithParameterFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 3;
    const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    auto convolution_1 = ngraph::builder::makeConvolution(parameters[0], ov::element::f32, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, pad_type, out_channels);
    convolution_1->set_friendly_name("convolution_1");

    auto bias_const_1 = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    bias_const_1->set_friendly_name("bias_const_1");
    auto bias_1 = ngraph::builder::makeEltwise(convolution_1, bias_const_1, ngraph::helpers::ADD);
    bias_1->set_friendly_name("bias_1");


    auto add = ngraph::builder::makeEltwise(bias_1, parameters[1], ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

std::shared_ptr<ov::Model> ConvAndAddWithParameterFunction::initReference() {
    if (input_shapes[0][0].get_length() == 1) {
        return initOriginal();
    }
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    const size_t bs_left = input_shapes[0][0].get_length();

    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, bs_left);

    const size_t out_channels = 3;
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 1, 1});
    const auto bias_const = ov::opset1::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    const std::string conv_name = "convolution";
    const std::string bias_name = "bias";

    ov::OutputVector concat_inputs(bs_left);
    for (size_t i = 0; i < bs_left; ++i) {
        std::string name_suffix = "_" + std::to_string(i);
        const auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                           weights_const,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::CoordinateDiff{0, 0},
                                                                           ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + name_suffix);
        const auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
        bias->set_friendly_name(bias_name + name_suffix);
        concat_inputs[i] = bias;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 0);
    concat->set_friendly_name(bias_name);

    const auto add = ngraph::builder::makeEltwise(concat, parameters[1], ngraph::helpers::ADD);
    add->set_friendly_name("add");

    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

ConvWithTransposeAndAddFunction::ConvWithTransposeAndAddFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithTransposeAndAddFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    const auto transpose_const_1 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_1 = std::make_shared<ov::opset1::Transpose>(convolution_1, transpose_const_1);
    transpose_1->set_friendly_name("transpose_1");

    auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    const auto transpose_const_2 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(convolution_2, transpose_const_2);
    transpose_2->set_friendly_name("transpose_2");

    auto add = ngraph::builder::makeEltwise(transpose_1, transpose_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

std::shared_ptr<ov::Model> ConvWithTransposeAndAddFunction::initReference() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);
    const std::string conv_name = "convolution";
    const std::string transpose_name = "transpose";
    const std::string add_name = "add";

    auto get_subgraph = [&](const size_t branch_idx, std::shared_ptr<ov::opset1::Parameter>& param) {
        const size_t out_channels = 30;
        const size_t in_channels = input_shapes[0][1].get_length();
        const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
        const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
        std::string conv_branch_name = conv_name + "_" + std::to_string(branch_idx);
        std::string transpose_branch_name = transpose_name + "_" + std::to_string(branch_idx);

        const size_t batch_size = param->get_partial_shape()[0].get_length();
        if (batch_size == 1) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(param,
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name);
            auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
            transpose->set_friendly_name(transpose_branch_name);
            return ov::OutputVector{transpose};
        }

        const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
        const auto split = std::make_shared<ov::opset1::Split>(param, split_axis, batch_size);
        ov::OutputVector res;
        for (size_t i = 0; i < batch_size; ++i) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name + "_" + std::to_string(i));
            auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
            transpose->set_friendly_name(transpose_branch_name + "_" + std::to_string(i));
            res.push_back(transpose);
        }
        return res;
    };

    const auto left_subgraph = get_subgraph(1, parameters[0]);
    const auto right_subgraph = get_subgraph(2, parameters[1]);
    const size_t out_batch = std::max(input_shapes[0][0].get_length(), input_shapes[1][0].get_length());

    ov::OutputVector concat_inputs(out_batch);
    for (size_t i = 0; i < out_batch; ++i) {
        auto add = ngraph::builder::makeEltwise(left_subgraph.size() == 1 ? left_subgraph[0] : left_subgraph[i],
                                                right_subgraph.size() == 1 ? right_subgraph[0] : right_subgraph[i],
                                                ngraph::helpers::ADD);
        add->set_friendly_name(add_name + "_" + std::to_string(i));
        concat_inputs[i] = add;
    }

    auto concat = std::make_shared<ov::opset1::Concat>(concat_inputs, 1);
    concat->set_friendly_name(add_name);
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvWithConcatFunction::ConvWithConcatFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithConcatFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    const auto transpose_const_1 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_1 = std::make_shared<ov::opset1::Transpose>(convolution_1, transpose_const_1);
    transpose_1->set_friendly_name("transpose_1");

    auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    const auto transpose_const_2 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(convolution_2, transpose_const_2);
    transpose_2->set_friendly_name("transpose_2");

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{transpose_1, transpose_2}, 1);
    concat->set_friendly_name("concat");
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

std::shared_ptr<ov::Model> ConvWithConcatFunction::initReference() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const size_t in_channels = input_shapes[0][1].get_length();
    std::string conv_name = "convolution";
    std::string transpose_name = "transpose";

    auto get_subgraph = [&](const size_t branch_idx, std::shared_ptr<ov::opset1::Parameter>& param) -> std::shared_ptr<ov::Node> {
        const auto weights_const = getRandomConst(ov::element::f32, {out_channels, in_channels, 3, 3});
        const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
        std::string conv_branch_name = conv_name + "_" + std::to_string(branch_idx);
        std::string transpose_branch_name = transpose_name + "_" + std::to_string(branch_idx);

        const size_t batch_size = param->get_partial_shape()[0].get_length();
        if (batch_size == 1) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(param,
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name);
            const auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
            transpose->set_friendly_name(transpose_branch_name);
            return transpose;
        }

        const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
        const auto split = std::make_shared<ov::opset1::Split>(param, split_axis, batch_size);
        ov::OutputVector concat_inputs(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                         weights_const,
                                                                         ov::Strides{1, 1},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::CoordinateDiff{0, 0},
                                                                         ov::Strides{1, 1});
            convolution->set_friendly_name(conv_branch_name + "_" + std::to_string(i));
            const auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
            transpose->set_friendly_name(transpose_branch_name + "_" + std::to_string(i));
            concat_inputs[i] = transpose;
        }

        const auto concat = ngraph::builder::makeConcat(concat_inputs, 1);
        concat->set_friendly_name(transpose_branch_name);
        return concat;
    };

    const auto subgraph_left = get_subgraph(1, parameters[0]);
    const auto subgraph_right = get_subgraph(2, parameters[1]);

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{subgraph_left, subgraph_right}, 1);
    concat->set_friendly_name("concat");
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}

ConvWithTransposeAddFunction::ConvWithTransposeAddFunction(
    const std::vector<ov::PartialShape>& input_shapes)
    : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithTransposeAddFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t in_channels = parameters[0]->get_partial_shape()[1].get_length();
    const auto weights_const = getRandomConst(ov::element::f32, {in_channels, in_channels, 1, 1});
    auto convolution = std::make_shared<ov::opset1::Convolution>(parameters[0],
                                                                 weights_const,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    convolution->set_friendly_name("convolution");

    const auto transpose_const_1 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_1 = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const_1);
    transpose_1->set_friendly_name("transpose_1");

    const auto transpose_const_2 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(parameters[1], transpose_const_2);
    transpose_2->set_friendly_name("transpose_2");

    auto add = ngraph::builder::makeEltwise(transpose_1, transpose_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

std::shared_ptr<ov::Model> ConvWithTransposeAddFunction::initReference() {
    if (input_shapes[0][0].get_length() == 1) {
        return initOriginal();
    }
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const std::string transpose_2_name = "transpose_2";
    const size_t in_channels = input_shapes[0][1].get_length();
    const auto weights_const = getRandomConst(ov::element::f32, {in_channels, in_channels, 1, 1});
    const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});

    const size_t batch_size = parameters[0]->get_partial_shape()[0].get_length();
    const auto split_axis = ov::opset1::Constant::create(ov::element::i32, {}, {0});
    const auto split = std::make_shared<ov::opset1::Split>(parameters[0], split_axis, batch_size);

    std::string conv_name = "convolution";
    std::string transpose_1_name = "transpose_1";

    ov::OutputVector concat_inputs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        auto convolution = std::make_shared<ov::opset1::Convolution>(split->output(i),
                                                                     weights_const,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1});
        convolution->set_friendly_name(conv_name + "_" + std::to_string(i));
        const auto transpose = std::make_shared<ov::opset1::Transpose>(convolution, transpose_const);
        transpose->set_friendly_name(transpose_1_name + "_" + std::to_string(i));
        concat_inputs[i] = transpose;
    }

    const auto concat = ngraph::builder::makeConcat(concat_inputs, 1);
    concat->set_friendly_name(transpose_1_name);

    const auto transpose_const_2 = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(parameters[1], transpose_const_2);
    transpose_2->set_friendly_name(transpose_2_name);

    auto add = ngraph::builder::makeEltwise(concat, transpose_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");

    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}
}  // namespace mixed_affinity
}  // namespace test
}  // namespace ov

