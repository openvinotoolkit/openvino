// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mixed_affinity_functions.hpp"

#include <ngraph_transformations/rt_info/optimal_batch_size.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/builders.hpp"
#include <ov_ops/type_relaxed.hpp>


namespace {
std::shared_ptr<ov::Node> getDefaultConv(const std::shared_ptr<ov::Node> data, const size_t out_channels) {
    const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;
    return ngraph::builder::makeConvolution(data, ov::element::f32, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, pad_type, out_channels);
}
} // namespace

void MixedAffinityFunctionBase::markup_model(const std::shared_ptr<ov::Model>& m, const BSMarkup& markup) {
    for (const auto& node : m->get_ordered_ops()) {
        auto it = markup.find(node->get_friendly_name());
        if (it != markup.end()) {
            ov::intel_cpu::set_optimal_bs(node, it->second);
        }
    }
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getOriginal(const BSMarkup& markup) {
    const auto original = initOriginal();
    markup_model(original, markup);
    return original;
}

std::shared_ptr<ov::Model> MixedAffinityFunctionBase::getReference(const BSMarkup& markup) {
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

    const auto bias_const = ov::opset8::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
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

    const auto transpose_const = ov::opset8::Constant::create(ov::element::i32, {4}, {1, 0, 2, 3});
    const auto transpose = std::make_shared<ov::opset8::Transpose>(convolution, transpose_const);
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

    const auto reshape_const = ov::opset8::Constant::create(ov::element::i32, {4}, {4, 30, 16, 16});
    const auto reshape = std::make_shared<ov::opset8::Reshape>(convolution, reshape_const, true);
    reshape->set_friendly_name("reshape");

    return std::make_shared<ov::Model>(ov::NodeVector{reshape}, parameters);
}

TwoConvAndAddFunction::TwoConvAndAddFunction(const std::vector<ov::PartialShape>& input_shapes) : MixedAffinityFunctionBase(input_shapes) {
    NGRAPH_CHECK(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> TwoConvAndAddFunction::initOriginal() {
    const auto parameters = ngraph::builder::makeDynamicParams(ov::element::f32, input_shapes);

    const size_t out_channels = 30;
    const auto convolution_1 = getDefaultConv(parameters[0], out_channels);
    convolution_1->set_friendly_name("convolution_1");

    const auto bias_const_1 = ov::opset8::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    bias_const_1->set_friendly_name("bias_const_1");
    const auto bias_1 = ngraph::builder::makeEltwise(convolution_1, bias_const_1, ngraph::helpers::ADD);
    bias_1->set_friendly_name("bias_1");

    const auto convolution_2 = getDefaultConv(parameters[1], out_channels);
    convolution_2->set_friendly_name("convolution_2");

    const auto bias_const_2 = ov::opset8::Constant::create(ov::element::f32, {1, out_channels, 1, 1}, {2.f});
    bias_const_2->set_friendly_name("bias_const_2");
    const auto bias_2 = ngraph::builder::makeEltwise(convolution_2, bias_const_2, ngraph::helpers::ADD);
    bias_2->set_friendly_name("bias_2");

    const auto add = ngraph::builder::makeEltwise(bias_1, bias_2, ngraph::helpers::ADD);
    add->set_friendly_name("add");
    return std::make_shared<ov::Model>(ov::NodeVector{add}, parameters);
}
