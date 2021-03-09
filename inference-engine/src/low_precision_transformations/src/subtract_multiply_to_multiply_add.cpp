// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/subtract_multiply_to_multiply_add.hpp"

#include <memory>
#include <string>
#include <vector>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/common/dequantization_op.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void SubtractMultiplyToMultiplyAddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

FakeQuantizeDequantization get(const std::shared_ptr<Node> node) {
    Output<Node> dataNode = node;

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = is_type<opset1::Constant>(
        dataNode.get_node_shared_ptr()->get_input_node_shared_ptr(1)) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode.get_node_shared_ptr()) :
        nullptr;
    std::shared_ptr<opset1::Constant> multiplyConstant;
    if (multiply != nullptr) {
        FakeQuantizeDequantization::fillDequantizationParams(multiply, multiplyConstant);
        dataNode = multiply->get_input_source_output(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = (dataNode.get_node_shared_ptr()->get_input_size() > 1ul)
        && is_type<opset1::Constant>(dataNode.get_node_shared_ptr()->get_input_node_ptr(1)) ?
            as_type_ptr<opset1::Subtract>(dataNode.get_node_shared_ptr()) :
            nullptr;
    std::shared_ptr<opset1::Convert> subtractConvert;
    std::shared_ptr<opset1::Constant> subtractConstant;
    if (subtract != nullptr) {
        FakeQuantizeDequantization::fillDequantizationParams(subtract, subtractConvert, subtractConstant);
        dataNode = subtract->get_input_source_output(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode.get_node_shared_ptr());
    if (convert != nullptr) {
        dataNode = convert->get_input_source_output(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, subtractConvert, subtractConstant, multiply, multiplyConstant);
}

bool SubtractMultiplyToMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    multiply = NetworkHelper::separateInStandaloneBranch(multiply);
    FakeQuantizeDequantization dequantization = get(multiply);

    const element::Type precisionBeforeDequantization = dequantization.convert == nullptr ?
        (dequantization.subtract == nullptr ?
            dequantization.multiply->get_input_element_type(0) :
            dequantization.subtract->get_input_element_type(0)) :
        dequantization.convert->get_input_element_type(0);

    const element::Type precisionAfterDequantization = dequantization.multiply->get_output_element_type(0);

    if (dequantization.empty()) {
        return false;
    }

    auto lastNew = dequantization.data;
    element::Type lastNewPrecision = precisionBeforeDequantization;
    std::shared_ptr<Node> lastPrevious = dequantization.multiply != nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.multiply) :
        dequantization.subtract;

    {
        const std::shared_ptr<Node> multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);

        lastNew = std::make_shared<op::TypeRelaxed<DequantizationMultiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{deqPrecision},
            ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(multiplyConstant, element::f32).get());

        if (dequantization.multiply != nullptr) {
            auto lastNewPtr = lastNew.get_node_shared_ptr();
            NetworkHelper::copyInfo(dequantization.multiply, lastNewPtr);
        }

        lastNewPrecision = deqPrecision;
    }

    if (dequantization.subtract != nullptr) {
        std::shared_ptr<Node> originalSubtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);

        std::shared_ptr<Node> subtractConstant = fold<opset1::Multiply>(
            fold<opset1::Multiply>(
                fold<opset1::Convert>(originalSubtractConstant, deqPrecision),
                std::make_shared<opset1::Constant>(deqPrecision, Shape{}, std::vector<float>{ -1.f })),
            fold<opset1::Convert>(dequantization.multiply->get_input_node_shared_ptr(1), deqPrecision));

        if (is_type<opset1::Constant>(subtractConstant)) {
            std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(subtractConstant);
            if (NetworkHelper::isScalarLike(constant)) {
                subtractConstant = NetworkHelper::toScalar(constant);
            }
        }

        lastNew = std::make_shared<op::TypeRelaxed<DequantizationAdd>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{precisionAfterDequantization},
            ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(subtractConstant, element::f32).get());

        auto lastNewPtr = lastNew.get_node_shared_ptr();
        NetworkHelper::copyInfo(dequantization.subtract, lastNewPtr);

        lastNewPrecision = precisionAfterDequantization;
    } else {
        NetworkHelper::setOutDataPrecision(as_type_ptr<opset1::Multiply>(lastNew.get_node_shared_ptr()), precisionAfterDequantization);
    }

    const std::shared_ptr<Node> lastOriginal = dequantization.multiply == nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.subtract) :
        dequantization.multiply;
    const std::shared_ptr<Node> lastNewPtr = lastNew.get_node_shared_ptr();
    replace_node(lastOriginal, lastNewPtr);

    updateOutput(context, lastNewPtr, lastPrevious);
    return true;
}

bool SubtractMultiplyToMultiplyAddTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    FakeQuantizeDequantization dequantization = get(op);
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return false;
    }

    if (((dequantization.subtract == nullptr) || (!dequantization.subtract->get_rt_info().count("DEQUANTIZATION"))) &&
        (!dequantization.multiply->get_rt_info().count("DEQUANTIZATION"))) {
        return false;
    }

    return
        ((dequantization.subtract == nullptr) || FakeQuantizeDequantization::checkElementwise(dequantization.subtract)) &&
        FakeQuantizeDequantization::checkElementwise(dequantization.multiply);
}

bool SubtractMultiplyToMultiplyAddTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
