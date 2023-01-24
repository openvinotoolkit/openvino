// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity_functions.hpp"

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>

#include <ngraph_transformations/rt_info/optimal_batch_size.hpp>
#include <ngraph_transformations/rt_info/num_splits.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/builders.hpp"
#include <ov_ops/type_relaxed.hpp>


namespace {
std::shared_ptr<ov::Node> getDefaultConv(const ov::Output<ov::Node> data, const size_t out_channels) {
    const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    return ngraph::builder::makeConvolution(data, ov::element::f32, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, pad_type, out_channels);
}
} // namespace

MixedAffinityMarkup transformBSMarkup(const std::unordered_map<std::string, size_t>& markup) {
    MixedAffinityMarkup markup_with_nsplits;
    for (const auto& elem : markup)
        markup_with_nsplits[elem.first] = {elem.second, 0};
    return markup_with_nsplits;
}

void MixedAffinityFunctionBase::markup_model(const std::shared_ptr<ov::Model>& m, const MixedAffinityMarkup& markup) {
    for (const auto& node : m->get_ordered_ops()) {
        auto it = markup.find(node->get_friendly_name());
        if (it != markup.end()) {
            ov::intel_cpu::set_optimal_bs(node, it->second.first);
            if (it->second.second > 0)
                ov::intel_cpu::set_num_splits(node, it->second.second);
        }
    }
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getOriginal(const MixedAffinityMarkup& markup) {
    const auto original = initOriginal();
    markup_model(original, markup);
    return original;
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getReference(const MixedAffinityMarkup& markup) {
    const auto reference = initReference();
    markup_model(reference, markup);
    return reference;
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
    bias_const->set_friendly_name("bias_const");
    const auto bias = ngraph::builder::makeEltwise(convolution, bias_const, ngraph::helpers::ADD);
    bias->set_friendly_name("bias");

    return std::make_shared<ov::Model>(ov::NodeVector{bias}, parameters);
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

ConvWithReshapeFunction::ConvWithReshapeFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 1, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithReshapeFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const auto convolution = getDefaultConv(parameters[0], out_channels);
    convolution->set_friendly_name("convolution");

    const auto& conv_out_shape = convolution->get_output_partial_shape(0);
    std::vector<std::int64_t> requested_shape = {conv_out_shape[0].is_static() ? conv_out_shape[0].get_length() : 4,
                                                 conv_out_shape[1].get_length(),
                                                 -1};
    const auto reshape_const = ov::opset1::Constant::create(ov::element::i32, {3}, requested_shape);
    const auto reshape = std::make_shared<ov::opset1::Reshape>(convolution, reshape_const, true);
    reshape->set_friendly_name("reshape");

    return std::make_shared<ov::Model>(ov::NodeVector{reshape}, parameters);
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

    ov::ResultVector results{std::make_shared<ov::opset1::Result>(split->output(0)), std::make_shared<ov::opset1::Result>(relu_2)};
    return std::make_shared<ov::Model>(results, parameters);
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

ConvWithTransposeAndAddFunction::ConvWithTransposeAndAddFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithTransposeAndAddFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_1 = std::make_shared<ov::opset1::Transpose>(convolution_1, transpose_const);
    transpose_1->set_friendly_name("transpose_1");

    auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(convolution_2, transpose_const);
    transpose_2->set_friendly_name("transpose_2");

    auto add = ngraph::builder::makeEltwise(transpose_1, transpose_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}

ConvWithConcatFunction::ConvWithConcatFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> ConvWithConcatFunction::initOriginal() {
    auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    size_t out_channels = 30;
    auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    const auto transpose_const = ov::opset1::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose_1 = std::make_shared<ov::opset1::Transpose>(convolution_1, transpose_const);
    transpose_1->set_friendly_name("transpose_1");

    auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    const auto transpose_2 = std::make_shared<ov::opset1::Transpose>(convolution_2, transpose_const);
    transpose_2->set_friendly_name("transpose_2");

    auto concat = std::make_shared<ov::opset1::Concat>(ov::OutputVector{transpose_1, transpose_2}, 1);
    concat->set_friendly_name("concat");
    return std::make_shared<ov::Model>(ov::NodeVector{concat}, parameters);
}
