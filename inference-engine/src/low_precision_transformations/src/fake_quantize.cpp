// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize.hpp"

#include <cmath>
#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::FakeQuantizeTransformation, "FakeQuantizeTransformation", 0);

FakeQuantizeTransformation::FakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "FakeQuantizeTransformation");
    this->register_matcher(m, callback);
}

bool FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<opset1::FakeQuantize> layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());
    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return false;
    }

    if (NetworkHelper::isFQByDynamicDimension(layer)) {
        return false;
    }

    bool wasHandled = false;
    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = layer;
    do {
        fakeQuantize = fuseElementwise(context, this, fakeQuantize);
        wasHandled = wasHandled || (fakeQuantize != nullptr);
    } while (fakeQuantize != nullptr);

    return wasHandled;
}

namespace fq {

static std::shared_ptr<Node> updateShape(std::shared_ptr<Node> constantOp, const PartialShape& targetShape) {
    const Shape shape = constantOp->get_output_shape(0);
    if ((shape.size() < static_cast<size_t>(targetShape.rank().get_length())) && (shape.size() > 1ul)) {
        constantOp = fold<opset1::Unsqueeze>(
            constantOp,
            std::make_shared<opset1::Constant>(ngraph::element::i32, Shape{ 1 }, std::vector<size_t>({ 0ul })));
    }
    return constantOp;
}

static std::shared_ptr<Node> getData(const std::shared_ptr<Node>& eltwise) {
    if (!is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(0))) {
        return eltwise->get_input_node_shared_ptr(0);
    }

    if (!is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(1))) {
        return eltwise->get_input_node_shared_ptr(1);
    }

    return nullptr;
}

static std::shared_ptr<opset1::Constant> getConstant(const std::shared_ptr<Node>& eltwise) {
    if (eltwise->get_input_size() != 2) {
        return nullptr;
    }

    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(1));
    if (constant != nullptr) {
        return constant;
    }

    return as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(0));
}

}  // namespace fq

bool FakeQuantizeTransformation::checkElementwise(const std::shared_ptr<Node>& eltwise) {
    const auto eltwiseInputPShape = eltwise->get_input_partial_shape(0);
    const auto eltwiseOutputPShape = eltwise->get_output_partial_shape(0);
    if (eltwiseInputPShape != eltwiseOutputPShape || eltwiseInputPShape.rank().is_dynamic() || eltwiseOutputPShape.rank().is_dynamic()) {
        return false;
    }

    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (constant == nullptr) {
        return false;
    }

    Shape shape = constant->get_output_shape(0);
    if ((!shape.empty()) && (shape_size(shape) != 1ul)) {
        if ((eltwiseOutputPShape.rank().get_length() - shape.size()) > 1) {
            return false;
        }

        if ((eltwiseOutputPShape.rank().get_length() - shape.size()) == 1ul) {
            shape.insert(shape.begin(), 1ul);
        }

        for (size_t i = 2ul; i < shape.size(); ++i) {
            if (shape[i] != 1ul) {
                return false;
            }
        }
    }

    return fq::getData(eltwise) != nullptr;
}

std::shared_ptr<opset1::FakeQuantize> FakeQuantizeTransformation::fuseElementwise(
    TransformationContext& context,
    MatcherPass* matcherPass,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    std::shared_ptr<Node> inputLowConst_f32 = foldConvert(fakeQuantize->get_input_node_shared_ptr(1), deqPrecision);
    std::shared_ptr<Node> inputHighConst_f32 = foldConvert(fakeQuantize->get_input_node_shared_ptr(2), deqPrecision);

    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (is_type<opset1::Multiply>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            foldConvert(constant, deqPrecision);

        const auto valueVec = as_type_ptr<opset1::Constant>(value)->cast_vector<float>();

        if (std::any_of(valueVec.cbegin(), valueVec.cend(), [](const float value) { return value <= 0.f; })) {
            return nullptr;
        }

        inputLowConst_f32 = fold<opset1::Divide>(inputLowConst_f32, value);
        inputHighConst_f32 = fold<opset1::Divide>(inputHighConst_f32, value);
        const auto resultLow = as_type_ptr<opset1::Constant>(inputLowConst_f32)->cast_vector<float>();
        const auto resultHigh = as_type_ptr<opset1::Constant>(inputHighConst_f32)->cast_vector<float>();
        if (std::any_of(resultLow.begin(), resultLow.end(), [](const float value){ return std::isinf(value); }) ||
            std::any_of(resultHigh.begin(), resultHigh.end(), [](const float value){ return std::isinf(value); })) {
            return nullptr;
        }

        inputLowConst_f32 = fq::updateShape(inputLowConst_f32, fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 =  fq::updateShape(inputHighConst_f32, fakeQuantize->get_output_partial_shape(0));
    } else if (is_type<opset1::Subtract>(eltwise) && checkElementwise(eltwise)) {
        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            foldConvert(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Add>(inputLowConst_f32, value), fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Add>(inputHighConst_f32, value), fakeQuantize->get_output_partial_shape(0));
    } else if (is_type<opset1::Add>(eltwise) && checkElementwise(eltwise)) {
        if (is_type<opset1::Convolution>(fq::getData(eltwise)) ||
            is_type<opset1::GroupConvolution>(fq::getData(eltwise)) ||
            is_type<opset1::ConvolutionBackpropData>(fq::getData(eltwise)) ||
            is_type<opset1::GroupConvolutionBackpropData>(fq::getData(eltwise))) {
            return nullptr;
        }

        const auto value = constant->get_output_element_type(0) == deqPrecision ?
            constant :
            foldConvert(constant, deqPrecision);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputLowConst_f32, value), fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputHighConst_f32, value), fakeQuantize->get_output_partial_shape(0));
    } else if (is_type<opset1::Convert>(eltwise)) {
        // issue #40611
        if ((eltwise->input(0).get_element_type() == element::i32) &&
            ((eltwise->output(0).get_element_type() == element::f16) || (eltwise->output(0).get_element_type() == element::f32))) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    const auto data = fq::getData(eltwise);
    const size_t outputIdx = NetworkHelper::getParentOutputIndex(data, eltwise);

    std::shared_ptr<opset1::FakeQuantize> newFakeQuantize = as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        data->output(outputIdx),
        inputLowConst_f32,
        inputHighConst_f32,
        foldConvert(fakeQuantize->input_value(3), deqPrecision),
        foldConvert(fakeQuantize->input_value(4), deqPrecision) }));

    matcherPass->register_new_node(newFakeQuantize);

    replace_node(fakeQuantize, newFakeQuantize);
    ngraph::copy_runtime_info({ fakeQuantize, eltwise }, newFakeQuantize);
    newFakeQuantize->set_friendly_name(fakeQuantize->get_friendly_name());
    NetworkHelper::cleanRunTimeInfo(newFakeQuantize);
    return newFakeQuantize;
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
