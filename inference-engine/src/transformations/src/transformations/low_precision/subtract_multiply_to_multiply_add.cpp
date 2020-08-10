// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/subtract_multiply_to_multiply_add.hpp"

#include <memory>
#include <string>
#include <vector>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void SubtractMultiplyToMultiplyAddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

FakeQuantizeDequantization get(const std::shared_ptr<Node> node) {
    std::shared_ptr<Node> dataNode = node;

    const std::shared_ptr<ngraph::opset1::Multiply> multiply = is_type<opset1::Constant>(dataNode->get_input_node_shared_ptr(1)) ?
        as_type_ptr<ngraph::opset1::Multiply>(dataNode) :
        nullptr;
    if (multiply != nullptr) {
        dataNode = multiply->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Subtract> subtract = (dataNode->get_input_size() > 1ul) && is_type<opset1::Constant>(dataNode->get_input_node_ptr(1)) ?
        as_type_ptr<opset1::Subtract>(dataNode) :
        nullptr;
    if (subtract != nullptr) {
        dataNode = subtract->get_input_node_shared_ptr(0);
    }

    const std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(dataNode);
    if (convert != nullptr) {
        dataNode = convert->get_input_node_shared_ptr(0);
    }

    return FakeQuantizeDequantization(dataNode, convert, subtract, multiply);
}

bool SubtractMultiplyToMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return false;
    }

    FakeQuantizeDequantization dequantization = get(multiply);

    // multiply operation is mandatory: subtract without multiply is part of new LPT operation set
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return false;
    }

    const element::Type precisionBeforeDequantization = dequantization.convert == nullptr ?
        (dequantization.subtract == nullptr ?
            dequantization.multiply->get_input_element_type(0) :
            dequantization.subtract->get_input_element_type(0)) :
        dequantization.convert->get_input_element_type(0);

    const element::Type precisionAfterDequantization = dequantization.subtract == nullptr ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);

    multiply = separateInStandaloneBranch(multiply);
    dequantization = get(multiply);
    if (dequantization.empty()) {
        return false;
    }

    std::shared_ptr<Node> lastNew = dequantization.data;
    element::Type lastNewPrecision = precisionBeforeDequantization;
    std::shared_ptr<Node> lastPrevious = dequantization.multiply != nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.multiply) :
        dequantization.subtract;

    const Shape shape = dequantization.multiply->get_input_shape(0);
    Shape constShape = std::vector<size_t>(shape.size(), 1ul);
    constShape[1] = shape[1];

    const size_t constShapeVolume = shape_size(constShape);

    {
        std::shared_ptr<Node> multiplyConstant = dequantization.multiply != nullptr ?
            dequantization.multiply->get_input_node_shared_ptr(1) :
            std::make_shared<opset1::Constant>(precisionAfterDequantization, constShape, std::vector<float>(constShapeVolume, 1));

        multiplyConstant = fold<opset1::Broadcast>(
            multiplyConstant,
            std::make_shared<opset1::Constant>(element::i32, Shape{ constShape.size() }, constShape));

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
                std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
                ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(multiplyConstant, element::f32).get());

            NetworkHelper::setOutDataPrecision(as_type_ptr<opset1::Multiply>(lastNew), precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<opset1::Multiply>(lastNew, multiplyConstant);
        }

        if (dequantization.multiply != nullptr) {
            copy_runtime_info(dequantization.multiply, lastNew);
            //NetworkHelper::copyInfo(dequantization.multiply, lastNew);
        }

        lastNewPrecision = precisionAfterDequantization;
    }

    {
        std::shared_ptr<Node> originalSubtractConstant = dequantization.subtract != nullptr ?
            dequantization.subtract->get_input_node_shared_ptr(1) :
            std::make_shared<opset1::Constant>(precisionAfterDequantization, constShape, std::vector<float>(constShapeVolume, 0));

        std::shared_ptr<Node> subtractConstant = fold<opset1::Multiply>(
            fold<opset1::Multiply>(
                fold<opset1::Convert>(originalSubtractConstant, precisionAfterDequantization),
                std::make_shared<opset1::Constant>(precisionAfterDequantization, Shape{}, std::vector<float>{ -1.f })),
            fold<opset1::Convert>(dequantization.multiply->get_input_node_shared_ptr(1), precisionAfterDequantization));

        {
            subtractConstant = fold<opset1::Broadcast>(
                subtractConstant,
                std::make_shared<opset1::Constant>(element::i32, Shape{ constShape.size() }, constShape));
        }

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<opset1::Add>>(
                std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
                ngraph::op::TemporaryReplaceOutputType(lastNew, element::f32).get(),
                ngraph::op::TemporaryReplaceOutputType(subtractConstant, element::f32).get());
            NetworkHelper::setOutDataPrecision(as_type_ptr<opset1::Add>(lastNew), precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<opset1::Add>(lastNew, subtractConstant);
        }

        if (dequantization.subtract != nullptr) {
            NetworkHelper::copyInfo(dequantization.subtract, lastNew);
        }

        lastNewPrecision = precisionAfterDequantization;
    }

    const std::shared_ptr<Node> lastOriginal = dequantization.multiply == nullptr ?
        std::dynamic_pointer_cast<Node>(dequantization.subtract) :
        dequantization.multiply;
    replace_node(lastOriginal, lastNew);

    updateOutput(context, lastNew, lastPrevious);
    return true;
}

bool SubtractMultiplyToMultiplyAddTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    FakeQuantizeDequantization dequantization = get(op);
    if (dequantization.empty() || is_type<opset1::FakeQuantize>(dequantization.data)) {
        return false;
    }

    // TODO: check if Convert & Subtract & Multiply are LPT dequantization operations

    if (op->get_output_shape(0).size() < 4ul) {
        return false;
    }

    // TODO: check if dequantization operations have appropriate Shape for ScaleShift
    auto isSupportedByScaleShift = [](const std::shared_ptr<Node> eltwise) -> bool {
        const ngraph::PartialShape constPartialShape = eltwise->get_input_partial_shape(1);
        if (constPartialShape.is_dynamic()) {
            return false;
        }

        const ngraph::Shape constShape = constPartialShape.to_shape();
        if ((constShape.size() == 0ul) || (constShape.size() == 1ul)) {
            return true;
        }

        if (constShape.size() < 4ul) {
            return false;
        }

        return shape_size(constShape) == constShape[1];
    };

    return
        ((dequantization.subtract == nullptr) || isSupportedByScaleShift(dequantization.subtract)) &&
        isSupportedByScaleShift(dequantization.multiply);
}

bool SubtractMultiplyToMultiplyAddTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
