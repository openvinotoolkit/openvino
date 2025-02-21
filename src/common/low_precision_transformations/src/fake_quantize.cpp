// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize.hpp"

#include <cmath>
#include <memory>
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FakeQuantizeTransformation::FakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FakeQuantizeTransformation);
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool FakeQuantizeTransformation::transform(ov::pass::pattern::Matcher &m) {
    const auto layer = ov::as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (!layer || !QuantizationDetails::outputLayoutIsSupported(layer)) {
        return false;
    }

    bool wasHandled = false;
    std::shared_ptr<opset1::FakeQuantize> fakeQuantize = layer;
    do {
        fakeQuantize = fuseElementwise(this, fakeQuantize, updatePrecisions);
        wasHandled = wasHandled || (fakeQuantize != nullptr);
    } while (fakeQuantize != nullptr);

    return wasHandled;
}

namespace fq {
namespace {

std::shared_ptr<Node> updateShape(std::shared_ptr<Node> constantOp, const PartialShape& targetShape) {
    assert(constantOp->get_output_partial_shape(0).is_static());
    const Shape shape = constantOp->get_output_shape(0);

    if ((shape.size() > 1ul) && (shape.size() < static_cast<size_t>(targetShape.rank().get_length()))) {
        constantOp = fold<opset1::Unsqueeze>(
            constantOp,
            std::make_shared<opset1::Constant>(ov::element::i32, Shape{ 1 }, std::vector<size_t>({ 0ul })));
    }
    return constantOp;
}

std::shared_ptr<Node> getDataNode(const std::shared_ptr<Node>& eltwise) {
    if (!ov::is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(0))) {
        return eltwise->get_input_node_shared_ptr(0);
    }

    if (!ov::is_type<opset1::Constant>(eltwise->get_input_node_shared_ptr(1))) {
        return eltwise->get_input_node_shared_ptr(1);
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> getConstant(const std::shared_ptr<Node>& eltwise) {
    if (eltwise->get_input_size() != 2) {
        return nullptr;
    }

    std::shared_ptr<opset1::Constant> constant = ov::as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(1));
    if (constant != nullptr) {
        return constant;
    }

    return ov::as_type_ptr<opset1::Constant>(eltwise->get_input_node_shared_ptr(0));
}

bool all_precisions_equal(const std::shared_ptr<Node>& node) {
    const auto& inputs = node->inputs();
    const auto first_input_precision = inputs.empty() ? element::dynamic : inputs[0].get_element_type();
    if (!inputs.empty()) {
        const auto first_input_precision = inputs[0].get_element_type();
        if (std::any_of(
            inputs.begin(),
            inputs.end(),
            [first_input_precision](const ov::Input<ov::Node>& input) {
                return input.get_element_type() != first_input_precision;
            })) {
            return false;
        }
    }

    const auto& outputs = node->outputs();
    if (!outputs.empty()) {
        const auto first_output_precision = outputs[0].get_element_type();
        if ((first_input_precision != element::dynamic) && (first_input_precision != first_output_precision)) {
            return false;
        }

        if (std::any_of(
            outputs.begin(),
            outputs.end(),
            [first_output_precision](const ov::Output<ov::Node>& output) {
                return output.get_element_type() != first_output_precision;
            })) {
            return false;
        }
    }

    return true;
}

}  // namespace
}  // namespace fq

bool FakeQuantizeTransformation::checkElementwise(const std::shared_ptr<Node>& eltwise) {
    const std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (constant == nullptr) {
        return false;
    }

    Shape shape = constant->get_shape();
    if (shape_size(shape) != 1ul) {
        const auto eltwiseInputPShape = eltwise->get_input_partial_shape(0);
        const auto eltwiseOutputPShape = eltwise->get_output_partial_shape(0);
        if (eltwiseInputPShape != eltwiseOutputPShape || eltwiseInputPShape.rank().is_dynamic() || eltwiseOutputPShape.rank().is_dynamic()) {
            return false;
        }

        while (eltwiseOutputPShape.size() > shape.size()) {
            shape.insert(shape.begin(), 1ul);
        }

        for (size_t i = 2ul; i < shape.size(); ++i) {
            if (shape[i] != 1ul) {
                return false;
            }
        }
    }

    return fq::getDataNode(eltwise) != nullptr;
}

std::shared_ptr<opset1::FakeQuantize> FakeQuantizeTransformation::fuseElementwise(
    MatcherPass* matcherPass,
    const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize,
    const bool updatePrecisions) {
    const std::shared_ptr<Node> eltwise = fakeQuantize->get_input_node_shared_ptr(0);

    if (!updatePrecisions && !fq::all_precisions_equal(eltwise)) {
        return nullptr;
    }

    if (!getAttribute<DisableCleanupAttribute>(eltwise).empty()) {
        return nullptr;
    }

    std::shared_ptr<Node> inputLowConst_f32 = foldConvert(fakeQuantize->input_value(1), element::f32);
    std::shared_ptr<Node> inputHighConst_f32 = foldConvert(fakeQuantize->input_value(2), element::f32);

    std::shared_ptr<opset1::Constant> constant = fq::getConstant(eltwise);
    if (ov::is_type<opset1::Multiply>(eltwise) && checkElementwise(eltwise)) {
        const auto value = foldConvert(constant, element::f32);

        const auto valueVec = ov::as_type_ptr<opset1::Constant>(value)->cast_vector<float>();

        if (std::any_of(valueVec.cbegin(), valueVec.cend(), [](const float value) { return value <= 0.f; })) {
            return nullptr;
        }

        inputLowConst_f32 = fold<opset1::Divide>(inputLowConst_f32, value);
        inputHighConst_f32 = fold<opset1::Divide>(inputHighConst_f32, value);
        if (!NetworkHelper::checkConstantNotInf(inputLowConst_f32) ||
            !NetworkHelper::checkConstantNotInf(inputHighConst_f32)) {
            return nullptr;
        }

        inputLowConst_f32 = fq::updateShape(inputLowConst_f32, fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 =  fq::updateShape(inputHighConst_f32, fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Subtract>(eltwise) && checkElementwise(eltwise)) {
        const auto value = foldConvert(constant, element::f32);

        inputLowConst_f32 = fq::updateShape(fold<opset1::Add>(inputLowConst_f32, value), fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Add>(inputHighConst_f32, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Add>(eltwise) && checkElementwise(eltwise) && !ov::marked_as_bias(eltwise)) {
        const auto value = foldConvert(constant, element::f32);
        inputLowConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputLowConst_f32, value), fakeQuantize->get_output_partial_shape(0));
        inputHighConst_f32 = fq::updateShape(fold<opset1::Subtract>(inputHighConst_f32, value), fakeQuantize->get_output_partial_shape(0));
    } else if (ov::is_type<opset1::Convert>(eltwise)) {
        // issue #40611
        if ((eltwise->get_input_element_type(0) == element::i32) &&
            ((eltwise->get_output_element_type(0) == element::f16) || (eltwise->get_output_element_type(0) == element::f32))) {
            return nullptr;
        }
    } else {
        return nullptr;
    }

    // issue #79980
    const auto data = eltwise->get_input_size() == 1ul ? eltwise->get_input_node_shared_ptr(0) : fq::getDataNode(eltwise);
    const size_t outputIdx = NetworkHelper::getParentOutputIndex(data, eltwise);

    const auto newFakeQuantize = ov::as_type_ptr<opset1::FakeQuantize>(fakeQuantize->clone_with_new_inputs({
        data->output(outputIdx),
        inputLowConst_f32,
        inputHighConst_f32,
        foldConvert(fakeQuantize->input_value(3), element::f32),
        foldConvert(fakeQuantize->input_value(4), element::f32) }));

    matcherPass->register_new_node(newFakeQuantize);

    replace_node(fakeQuantize, newFakeQuantize);
    ov::copy_runtime_info({ fakeQuantize, eltwise }, newFakeQuantize);
    newFakeQuantize->set_friendly_name(fakeQuantize->get_friendly_name());
    return newFakeQuantize;
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ov
