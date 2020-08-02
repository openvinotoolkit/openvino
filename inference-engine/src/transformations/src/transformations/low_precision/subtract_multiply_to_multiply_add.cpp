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

void SubtractMultiplyToMultiplyAddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();
    if (!canBeTransformed(context, multiply)) {
        return;
    }

    FakeQuantizeDequantization dequantization = get(multiply);

    // multiply operation is mandatory: subtract without multiply is part of new LPT operation set
    if (dequantization.empty() || (dequantization.multiply == nullptr)) {
        return;
    }

    const element::Type precisionBeforeDequantization = dequantization.convert == nullptr ?
        (dequantization.subtract == nullptr ?
            dequantization.multiply->get_input_node_ptr(0)->get_output_element_type(NetworkHelper::getInputIndex(
                dequantization.multiply->get_input_node_shared_ptr(0),
                dequantization.multiply)) :
            dequantization.subtract->get_input_node_ptr(0)->get_output_element_type(NetworkHelper::getInputIndex(
                dequantization.subtract->get_input_node_shared_ptr(0),
                dequantization.subtract))) :
        dequantization.convert->get_input_node_ptr(0)->get_output_element_type(NetworkHelper::getInputIndex(
            dequantization.convert->get_input_node_shared_ptr(0),
            dequantization.convert));

    const element::Type precisionAfterDequantization = dequantization.subtract == nullptr ?
        dequantization.multiply->get_output_element_type(0) :
        dequantization.subtract->get_output_element_type(0);


    multiply = separateInStandaloneBranch(multiply);
    dequantization = get(multiply);
    if (dequantization.empty()) {
        return;
    }

    std::shared_ptr<Node> lastNew = dequantization.data;
    element::Type lastNewPrecision = precisionBeforeDequantization;
    std::shared_ptr<Node> lastPrevious = dequantization.multiply != nullptr ?
        as_type_ptr<Node>(dequantization.multiply) :
        dequantization.subtract;

    Shape constShape = dequantization.multiply != nullptr ?
        dequantization.multiply->get_input_node_shared_ptr(1)->get_output_shape(0) :
        dequantization.subtract->get_input_node_shared_ptr(1)->get_output_shape(0);

    size_t constShapeVolume = 1ul;
    if (!constShape.empty()) {
        for (size_t i = 0; i < constShape.size(); ++i) {
            constShapeVolume *= constShape[i];
        }
    }

    const bool convertPowerToScaleShift = true;
    if ((constShapeVolume == 1ul) && convertPowerToScaleShift) {
        const Shape shape = dequantization.multiply->get_input_node_ptr(0)->get_output_shape(NetworkHelper::getInputIndex(
            dequantization.multiply->get_input_node_shared_ptr(0),
            dequantization.multiply));
        constShape = std::vector<size_t>(shape.size(), 1ul);
        constShape[1] = shape[1];
    }

    {
        std::shared_ptr<Node> multiplyConstant = dequantization.multiply != nullptr ?
            dequantization.multiply->get_input_node_shared_ptr(1) :
            std::make_shared<opset1::Constant>(precisionAfterDequantization, constShape, std::vector<float>(constShapeVolume, 1));

        if (convertPowerToScaleShift) {
            multiplyConstant = fold<opset1::Broadcast>(
                multiplyConstant,
                std::make_shared<opset1::Constant>(element::i32, Shape{ constShape.size() }, constShape));
        }

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(lastNew, multiplyConstant);
            NetworkHelper::setOutDataPrecision(lastNew, precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<opset1::Multiply>(lastNew, multiplyConstant);
        }

        if (dequantization.multiply != nullptr) {
            NetworkHelper::copyInfo(dequantization.multiply, lastNew);
        }

        lastNewPrecision = precisionAfterDequantization;
    }

    if ((dequantization.subtract != nullptr) || (convertPowerToScaleShift && (dequantization.subtract == nullptr))) {
        std::shared_ptr<Node> originalSubtractConstant = dequantization.subtract != nullptr ?
            dequantization.subtract->get_input_node_shared_ptr(1) :
            std::make_shared<opset1::Constant>(precisionAfterDequantization, constShape, std::vector<float>(constShapeVolume, 0));

        std::shared_ptr<Node> subtractConstant = fold<opset1::Multiply>(
            fold<opset1::Multiply>(
                fold<opset1::Convert>(originalSubtractConstant, precisionAfterDequantization),
                std::make_shared<opset1::Constant>(precisionAfterDequantization, Shape{}, std::vector<float>{ -1.f })),
            fold<opset1::Convert>(dequantization.multiply->get_input_node_shared_ptr(1), precisionAfterDequantization));

        if (convertPowerToScaleShift) {
            subtractConstant = fold<opset1::Broadcast>(
                subtractConstant,
                std::make_shared<opset1::Constant>(element::i32, Shape{ constShape.size() }, constShape));
        }

        if (lastNewPrecision != precisionAfterDequantization) {
            lastNew = std::make_shared<op::TypeRelaxed<opset1::Add>>(lastNew, subtractConstant);
            NetworkHelper::setOutDataPrecision(lastNew, precisionAfterDequantization);
        } else {
            lastNew = std::make_shared<opset1::Add>(lastNew, subtractConstant);
        }

        if (dequantization.subtract != nullptr) {
            NetworkHelper::copyInfo(dequantization.subtract, lastNew);
        }

        lastNewPrecision = precisionAfterDequantization;
    }

    const std::shared_ptr<Node> lastOriginal = dequantization.multiply == nullptr ?
        as_type_ptr<Node>(dequantization.subtract) :
        dequantization.multiply;
    replace_node(lastOriginal, lastNew);

    updateOutput(context, lastNew, lastPrevious);
}

bool SubtractMultiplyToMultiplyAddTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    return op->get_output_shape(0).size() >= 4;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
