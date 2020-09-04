// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/reshape.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void ReshapeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Reshape>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void reshapeDequantizationConstant(const std::shared_ptr<opset1::Reshape>& reshape) {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(reshape, 0);
    if (dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0).size() > 1ul) {
        auto replaceConstant = [](const std::shared_ptr<opset1::Reshape>& reshape, const std::shared_ptr<Node>& op) {
            // Original Reshape operation is used to update operation Constant.
            // But original Reshape operation output data shape constant should be changed before reshape.

            // simple broadcast operation Constant shape to shape on activations
            auto newOperationConstantShape = op->input(1).get_shape();
            auto const reshapeInputShape = reshape->input(0).get_shape();
            if ((reshapeInputShape.size() - newOperationConstantShape.size()) == 1ul) {
                newOperationConstantShape.insert(newOperationConstantShape.begin(), 1ul);
            }
            const std::shared_ptr<opset1::Constant> originalConstant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(1));
            const std::shared_ptr<opset1::Constant> newOperationConstant = std::make_shared<opset1::Constant>(
                op->input(1).get_element_type(),
                newOperationConstantShape,
                originalConstant->cast_vector<float>());

            // update Reshape constant
            const std::vector<int> reshapeConstValues = as_type_ptr<opset1::Constant>(reshape->get_input_node_shared_ptr(1))->cast_vector<int>();
            std::vector<int> newReshapeConstValues(reshapeConstValues);
            for (int i = newReshapeConstValues.size() - 1; i >= 0; --i) {
                if (newOperationConstantShape.size() <= i) {
                    newReshapeConstValues[i] = 1;
                } else if (newOperationConstantShape[i] == 1ul) {
                    // not used dimension
                    newReshapeConstValues[i] = 1;
                } else {
                    break;
                }
            }

            const std::shared_ptr<opset1::Constant> newReshapeConstant = std::make_shared<opset1::Constant>(
                reshape->input(1).get_element_type(),
                Shape({ newReshapeConstValues.size() }),
                newReshapeConstValues);

            const std::shared_ptr<Node> resultConstant = fold<opset1::Reshape>(
                newOperationConstant,
                newReshapeConstant,
                reshape->get_special_zero());

            replace_node(op->get_input_node_shared_ptr(1), resultConstant);
        };

        if (dequantization.subtract != nullptr) {
            replaceConstant(reshape, dequantization.subtract);
        }

        if (dequantization.multiply != nullptr) {
            replaceConstant(reshape, dequantization.multiply);
        }
    }
}

bool ReshapeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::Reshape> reshape = as_type_ptr<opset1::Reshape>(m.get_match_root());
    if ((reshape == nullptr) || (!canBeTransformed(context, reshape))) {
        return false;
    }

    reshape = as_type_ptr<opset1::Reshape>(separateInStandaloneBranch(reshape));
    reshapeDequantizationConstant(reshape);
    moveDequantizationAfter(context, reshape, NetworkHelper::getDequantization(reshape, 0), false);
    return true;
}

bool ReshapeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

size_t getLastNotBroadcastedChannel(const Shape& shape) {
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (shape[i] != 1ul) {
            return i;
        }
    }
    return 0;
}

size_t getFirstChangedChannel(const Shape& shape1, const Shape& shape2) {
    const size_t minSize = std::min(shape1.size(), shape2.size());
    size_t i = 0;
    for (; i < minSize; ++i) {
        if (shape1[i] != shape2[i]) {
            return i;
        }
    }
    return i;
}

bool ReshapeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op);
    if (dequantization.empty()) {
        return false;
    }


    const Shape inputShape = op->get_input_shape(0);
    const Shape outputShape = op->get_output_shape(0);
    // TODO: story 38439
    if ((inputShape[0] != outputShape[0]) || (inputShape[1] != outputShape[1])) {
        return false;
    }

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtract->get_input_node_ptr(1)->get_output_shape(0);
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0);
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return true;
    }

    Shape subtractShapeWithBatch = subtractShape;
    if ((dequantization.subtract != nullptr) && (subtractShapeWithBatch.size() < inputShape.size())) {
        subtractShapeWithBatch.insert(subtractShapeWithBatch.begin(), inputShape[0]);
    }

    Shape multiplyShapeWithBatch = multiplyShape;
    if ((dequantization.multiply != nullptr) && (multiplyShapeWithBatch.size() < inputShape.size())) {
        multiplyShapeWithBatch.insert(multiplyShapeWithBatch.begin(), inputShape[0]);
    }

    return canBeTransformed(subtractShapeWithBatch, multiplyShapeWithBatch, inputShape, outputShape);
}

bool ReshapeTransformation::canBeTransformed(
    const ngraph::Shape& subtractShape,
    const ngraph::Shape& multiplyShape,
    const ngraph::Shape& inputShape,
    const ngraph::Shape& outputShape) {
    const size_t lastNotBroadcastedChannel = std::max(getLastNotBroadcastedChannel(subtractShape), getLastNotBroadcastedChannel(multiplyShape));
    const size_t firstChangedChannel = getFirstChangedChannel(inputShape, outputShape);
    if (lastNotBroadcastedChannel >= firstChangedChannel) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
