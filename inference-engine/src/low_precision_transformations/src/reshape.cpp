// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reshape.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ReshapeTransformation, "ReshapeTransformation", 0);

ReshapeTransformation::ReshapeTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Reshape>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ReshapeTransformation");
    this->register_matcher(m, callback);
}

void reshapeDequantizationConstant(const std::shared_ptr<opset1::Reshape>& reshape) {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(reshape, 0);
    if (dequantization.multiplyConstant->get_shape().size() > 1ul) {
        // Reshape Subtract or Multiply operation Constant.
        //    1. modify reshape parameters to avoid reshape by spatial dimensions
        //    2. broadcast element-wise constant if channels are changed
        //    3. reshape element-wise constant with modified reshape parameters
        auto replaceConstant = [](const std::shared_ptr<opset1::Reshape>& reshape, const std::shared_ptr<Node>& op) {
            const size_t constantIndex = as_type<ngraph::opset1::Constant>(op->get_input_node_ptr(1)) ? 1 : 0;
            const auto originalConstant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(constantIndex));
            const auto constantShape = originalConstant->get_shape();

            // reshape for element-wise constant is not required
            if (shape_size(constantShape) == 1ul) {
                if (constantShape.size() > 1ul) {
                    const Shape newConstShape = Shape(reshape->get_output_partial_shape(0).rank().get_length(), 1ul);
                    const auto newConstant = opset1::Constant::create(
                        originalConstant->get_element_type(), newConstShape, originalConstant->cast_vector<float>());
                    replace_node(op->get_input_node_shared_ptr(constantIndex), newConstant);
                }

                return;
            }

            // simple broadcast operation Constant shape to shape on activations
            auto newOperationConstantShape = constantShape;
            auto const reshapeInputPShape = reshape->get_input_partial_shape(0);
            PartialShape newOperationConstantBroadcastedShape(reshapeInputPShape);
            newOperationConstantBroadcastedShape[0] = 1ul;

            if ((reshapeInputPShape.rank().get_length() - newOperationConstantShape.size()) == 1ul) {
                newOperationConstantShape.insert(newOperationConstantShape.begin(), 1ul);
            }
            const std::shared_ptr<opset1::Constant> newOperationConstant = std::make_shared<opset1::Constant>(
                op->input(constantIndex).get_element_type(),
                newOperationConstantShape,
                originalConstant->cast_vector<float>());

            // reshape -1 value handling
            auto getOverallValue = [](const Shape& shape, const std::vector<int>& reshapeValues, const bool specialZero) -> size_t {
                size_t overallValue = shape_size(shape);
                for (size_t i = 0; i < reshapeValues.size(); ++i) {
                    auto reshapeValue = reshapeValues[i];
                    if ((reshapeValue == 1ul) || (reshapeValue == -1) || ((reshapeValue == 0ul) && !specialZero)) {
                        continue;
                    }

                    if ((reshapeValue == 0ul) && specialZero) {
                        reshapeValue = shape[i];
                    }

                    overallValue = overallValue / reshapeValue;
                }
                return overallValue;
            };

            // modify reshape constant for element-wise constant reshape
            // element-wise constant doesn't have spatial dimensions, as result we should remove spatial dimensions from reshape parameters
            const std::vector<int> reshapeConstValues = as_type_ptr<opset1::Constant>(reshape->get_input_node_shared_ptr(1))->cast_vector<int>();

            size_t overallValue = 0;
            for (size_t i = 0; i < reshapeConstValues.size(); ++i) {
                if (reshapeConstValues[i] == -1) {
                    overallValue = getOverallValue(
                        reshapeInputPShape.to_shape(),
                        reshapeConstValues,
                        as_type_ptr<opset1::Reshape>(reshape)->get_special_zero());
                    break;
                }
            }

            std::vector<int> newReshapeConstValues(reshapeConstValues);
            for (int i = static_cast<int>(newReshapeConstValues.size() - 1); i >= 0; --i) {
                if (static_cast<int64_t>(newOperationConstantShape.size()) <= i) {
                    // new dimension was added
                    newReshapeConstValues[i] = 1;
                } else if (newOperationConstantShape[i] == 1ul) {
                    // keep the same
                    newReshapeConstValues[i] = 1;
                } else if (newReshapeConstValues[i] == -1) {
                    // modified reshape parameters are different, but value instead '-1' has to be equal as original reshape
                    newReshapeConstValues[i] = overallValue;
                }
            }

            const std::shared_ptr<opset1::Constant> newReshapeConstant = std::make_shared<opset1::Constant>(
                reshape->input(1).get_element_type(),
                Shape({ newReshapeConstValues.size() }),
                newReshapeConstValues);

            // if channels are different then broadcast spatial dimensions to reshape channels correctly
            // limitation which has to be covered by canBeTransformed:
            //    1. spatial dimensions have to be absent or equal to 1 after reshape
            //    2. only second dimension can be changed

            const bool shouldBroadcast = (shape_size(newReshapeConstValues) != 1ul) && (reshapeConstValues[1] != 0) &&
                (((reshapeConstValues[1] != -1) &&
                    (static_cast<int64_t>(newOperationConstantShape[1]) != reshapeConstValues[1])) ||
                ((reshapeConstValues[1] == -1) &&
                    (newOperationConstantShape[1] != overallValue)));

            const std::shared_ptr<Node> broadcastedConstant = shouldBroadcast ?
                fold<opset1::Broadcast>(
                    newOperationConstant,
                    std::make_shared<opset1::Constant>(
                        element::i32,
                        Shape({static_cast<size_t>(newOperationConstantBroadcastedShape.rank().get_length())}),
                        // TODO: investigate behaviour
                        newOperationConstantBroadcastedShape.to_shape())) :
                newOperationConstant;

            const std::shared_ptr<Node> resultConstant = fold<opset1::Reshape>(
                broadcastedConstant,
                newReshapeConstant,
                reshape->get_special_zero());

            replace_node(op->get_input_node_shared_ptr(constantIndex), resultConstant);
        };

        if (dequantization.subtract != nullptr) {
            replaceConstant(reshape, dequantization.subtract);
        }

        if (dequantization.multiply != nullptr) {
            replaceConstant(reshape, dequantization.multiply);
        }
    }
}

bool ReshapeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<opset1::Reshape> reshape = as_type_ptr<opset1::Reshape>(m.get_match_root());
    if (NetworkHelper::isConstantPath(reshape)) {
        return false;
    }

    if (!canBeTransformed(context, reshape)) {
        return false;
    }

    reshape = as_type_ptr<opset1::Reshape>(NetworkHelper::separateInStandaloneBranch(reshape));
    reshapeDequantizationConstant(reshape);
    moveDequantizationAfter(context, reshape, NetworkHelper::getDequantization(reshape, 0), false);
    return true;
}

bool ReshapeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

size_t getLastNotBroadcastedChannel(const Shape& shape) {
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] != 1ul) {
            return i;
        }
    }
    return 0;
}

size_t getFirstChangedChannel(const PartialShape& shape1, const PartialShape& shape2) {
    const size_t minSize = std::min(shape1.rank().get_length(), shape2.rank().get_length());
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

    // TODO: LPT: to support current flow: #58269
    //if (((dequantization.subtractConstant != nullptr) && NetworkHelper::isScalarLike(dequantization.subtractConstant)) ||
    //    ((dequantization.multiplyConstant != nullptr) && NetworkHelper::isScalarLike(dequantization.multiplyConstant))) {
    //    return true;
    //}

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtractConstant->get_shape();
    Shape subtractShapeWithBatch = subtractShape;
    const PartialShape inputPShape = op->get_input_partial_shape(0);
    if (inputPShape.rank().is_dynamic()) {
        return false;
    }

    const size_t inputRank = inputPShape.rank().get_length();

    if ((dequantization.subtract != nullptr) &&
        (subtractShapeWithBatch.size() > 1ul) &&
        (subtractShapeWithBatch.size() < inputRank)) {
        subtractShapeWithBatch.insert(subtractShapeWithBatch.begin(), 1ul);
    }

    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiply->input(1).get_shape();
    Shape multiplyShapeWithBatch = multiplyShape;
    if ((dequantization.multiply != nullptr) &&
        (multiplyShapeWithBatch.size() > 1ul) &&
        (multiplyShapeWithBatch.size() < inputRank)) {
        multiplyShapeWithBatch.insert(multiplyShapeWithBatch.begin(), 1ul);
    }

    const PartialShape outputPShape = op->get_output_partial_shape(0);
    // if we have per-channel dq, dynamic shape, and "-1" reshape value - don't transform
    if (outputPShape.is_dynamic() && (shape_size(subtractShape) > 1ul || shape_size(multiplyShape) > 1ul)) {
        const auto reshapeConstant = as_type_ptr<opset1::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<int>();
        if (std::any_of(reshapeConstant.cbegin(), reshapeConstant.cend(), [](const int value) { return value == -1; })) {
            return false;
        }
    }

    return canBeTransformed(subtractShapeWithBatch, multiplyShapeWithBatch, inputPShape, outputPShape);
}

size_t getChannelVolume(const PartialShape& shape) {
    size_t volume = 1ul;
    for (int i = 2; i < shape.rank().get_length(); ++i) {
        volume = volume * shape[i].get_length();
    }
    return volume;
}

bool ReshapeTransformation::canBeTransformed(
    const ngraph::Shape& subtractShape,
    const ngraph::Shape& multiplyShape,
    const ngraph::PartialShape& inputShape,
    const ngraph::PartialShape& outputShape) {
    const size_t inputRank = inputShape.rank().get_length();
    const size_t outputRank = outputShape.rank().get_length();

    if ((inputRank < 2ul) || (outputRank < 2ul) || (inputShape[0] != outputShape[0])) {
        return false;
    }

    // TODO: story 38439
    if ((inputRank == 4ul) && (outputRank == 2ul)) {
        auto checkSpatialDimensions = [](const Shape& dequantizationConstShape) {
            for (size_t i = (dequantizationConstShape.size() - 2); i < dequantizationConstShape.size(); ++i) {
                if (dequantizationConstShape[i] != 1ul) {
                    return false;
                }
            }
            return true;
        };

        if (((subtractShape.size() >= 3ul) && (!checkSpatialDimensions(subtractShape))) ||
            ((multiplyShape.size() >= 3ul) && (!checkSpatialDimensions(multiplyShape)))) {
            return false;
        }

        if (inputRank > 1ul) {
            if (inputShape[1].is_dynamic()) {
                return false;
            }
        } else {
            if (inputShape[0].is_dynamic()) {
                return false;
            }
        }

        if (outputRank > 1ul) {
            if (outputShape[1].is_dynamic()) {
                return false;
            }
        } else {
            if (outputShape[0].is_dynamic()) {
                return false;
            }
        }

        // custom validation for Layout::NCHW => Layout::NC
        const size_t inputChannelsCount = inputRank > 1ul ? inputShape[1].get_length() : inputShape[0].get_length();
        const size_t outputChannelsCount = outputRank > 1ul ? outputShape[1].get_length() : outputShape[0].get_length();
        for (size_t i = 2; i < inputRank; ++i) {
            if (inputShape[i].is_dynamic()) {
                return false;
            }
        }

        if ((inputShape[0] != outputShape[0]) || ((inputChannelsCount * getChannelVolume(inputShape)) != outputChannelsCount)) {
            return false;
        }
    } else {
        if (ngraph::shape_size(subtractShape) > 1 || ngraph::shape_size(multiplyShape) > 1) {
            for (size_t i = 0; i < 2ul; ++i) {
                if (inputShape[i] != outputShape[i]) {
                    return false;
                }
            }
        }

        const size_t lastNotBroadcastedChannel = std::max(getLastNotBroadcastedChannel(subtractShape), getLastNotBroadcastedChannel(multiplyShape));
        const size_t firstChangedChannel = getFirstChangedChannel(inputShape, outputShape);
        if (lastNotBroadcastedChannel >= firstChangedChannel) {
            return false;
        }
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
