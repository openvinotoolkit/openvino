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

void reshapeDequantizationConstant(std::shared_ptr<Node>& reshape) {
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(reshape, 0);
    if (dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0).size() > 1ul) {
        // TODO: refactor: use constant instead op
        auto replaceConstant = [](const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& op) {
            std::vector<size_t> resultShape(reshape->get_output_shape(0).size(), 1ul);

            std::vector<size_t> actualShape = op->get_input_node_ptr(1)->get_output_shape(0);
            const size_t maxIndex = std::min(resultShape.size(), actualShape.size());
            for (size_t i = 0; i < maxIndex; ++i) {
                resultShape[i] = actualShape[i];
            }

            std::shared_ptr<Node> reshapeConstant = fold_reshape<opset1::Reshape>(
                op->get_input_node_shared_ptr(1),
                std::make_shared<opset1::Constant>(element::i64, Shape{ resultShape.size() }, resultShape),
                false);
            replace_node(op->get_input_node_shared_ptr(1), reshapeConstant);
        };

        if (dequantization.subtract != nullptr) {
            replaceConstant(reshape, dequantization.subtract);
        }

        if (dequantization.multiply != nullptr) {
            replaceConstant(reshape, dequantization.multiply);
        }
    }
}

void ReshapeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> reshape = m.get_match_root();
    if (!canBeTransformed(context, reshape)) {
        removeConvertIfPossible(context, reshape);
        return;
    }

    reshape = separateInStandaloneBranch(reshape);

    reshapeDequantizationConstant(reshape);
    moveDequantizationAfter(context, reshape, NetworkHelper::getDequantization(reshape, 0), false);
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
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, 0);

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtract->get_input_node_ptr(1)->get_output_shape(0);
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0);
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return true;
    }

    const size_t index = NetworkHelper::getInputIndex(op->get_input_node_shared_ptr(0), op);
    const auto inputShape = op->get_input_node_shared_ptr(0)->get_output_shape(index);
    const auto outputShape = op->get_output_shape(0);

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
