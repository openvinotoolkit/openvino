// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reshape.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReshapeTransformation::ReshapeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ReshapeTransformation);
    auto input = ov::pass::pattern::any_input();
    auto mul_const_m = pattern::wrap_type<ov::opset1::Constant>();
    auto mul_m = pattern::wrap_type<ov::opset1::Multiply>({ input, mul_const_m });
    auto reshape_pattern_const = pattern::wrap_type<ov::opset1::Constant>();
    auto reshape_pattern_nonconst = ov::pass::pattern::any_input();
    auto reshape_pattern = std::make_shared<pass::pattern::op::Or>(OutputVector{ reshape_pattern_const, reshape_pattern_nonconst });
    auto matcher = pattern::wrap_type<ov::opset1::Reshape>({ mul_m, reshape_pattern });

    ov::graph_rewrite_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        // we can propagate only per-tensor dq through reshape with non-const reshape_pattern
        const auto& pattern_map = m.get_pattern_value_map();
        if (pattern_map.count(reshape_pattern_nonconst)) {
            const auto mul_const = as_type_ptr<ov::opset1::Constant>(pattern_map.at(mul_const_m).get_node_shared_ptr());
            if (!mul_const || ov::shape_size(mul_const->get_shape()) != 1) {
                return false;
            }
        }

        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

namespace {

void reshapeDequantizationConstant(const std::shared_ptr<ov::opset1::Reshape>& reshape, const std::vector<ov::element::Type>& defaultPrecisions) {
    // Reshape dequantization operation Constant.
    //    1. Calculate result dequantization Constant shape for broadcast based on original dequantization Constant shape and Reshape output.
    //    For example: dequantization shape {1, 3, 1, 1}, output Reshape shape {1, 12, 3, 3}, result for broadcast: {1, 3, 4, 1},
    //    where '4' calculated for temporary broadcast before reshape.
    //    2. Broadcast dequantization Constant, if channels are changed
    //    3. Reshape and replace
    auto replaceConstant = [](const std::shared_ptr<ov::opset1::Reshape>& reshape, const std::shared_ptr<ov::opset1::Constant>& originalConstant) {
        // reshape for element-wise constant is not required
        auto constantShape = originalConstant->get_shape();
        if (NetworkHelper::isScalarLike(originalConstant)) {
            if (!constantShape.empty()) {
                const auto newConstant = NetworkHelper::toScalar(originalConstant);
                replace_node(originalConstant, newConstant);
            }
            return;
        }

        auto const reshapeInputRank = reshape->get_input_partial_shape(0).rank();
        assert(reshapeInputRank.is_static());
        if (constantShape.size() > 1ul) {
            while (constantShape.size() < static_cast<size_t>(reshapeInputRank.get_length())) {
                constantShape.insert(constantShape.begin(), 1ul);
            }
        }

        const auto reshapeOutputPShape = reshape->get_output_partial_shape(0);
        const auto reshapeOutputRank = reshapeOutputPShape.rank();
        assert(reshapeOutputRank.is_static());
        assert(reshapeOutputRank.get_length() >= 2);
        assert(reshapeOutputPShape[1].is_static());
        assert(static_cast<size_t>(reshapeOutputPShape[1].get_length()) >= constantShape[1]);
        assert(reshapeOutputPShape[1].get_length() % constantShape[1] == 0);
        const size_t dimensionsToBroadcast = reshapeOutputPShape[1].get_length() / constantShape[1];
        if (dimensionsToBroadcast == 0ul) {
            return;
        }

        auto getBCastedConst = [](const std::shared_ptr<ov::opset1::Constant>& constant, size_t dimensionsToBroadcast) -> std::shared_ptr<Node> {
            if (dimensionsToBroadcast == 1ul) {
                return constant;
            }

            Shape newOperationConstantBroadcastedShape = constant->get_shape();
            // add dimensions to broadcast values
            if (newOperationConstantBroadcastedShape.size() == 2ul) {
                newOperationConstantBroadcastedShape[0] = dimensionsToBroadcast;
            } else {
                newOperationConstantBroadcastedShape[2] = dimensionsToBroadcast;
            }

            const auto targetShapeConstant = ov::opset1::Constant::create(
                element::i32,
                Shape{ newOperationConstantBroadcastedShape.size() },
                newOperationConstantBroadcastedShape);

            return fold<ov::opset1::Broadcast>(constant, targetShapeConstant);
        };

        const std::shared_ptr<Node> broadcastedConstant = getBCastedConst(originalConstant, dimensionsToBroadcast);

        std::vector<int> newReshapeConstValues(reshapeOutputRank.get_length(), 1ul);
        newReshapeConstValues[1] = static_cast<int>(reshapeOutputPShape[1].get_length());
        const std::shared_ptr<ov::opset1::Constant> newReshapeConstant = std::make_shared<ov::opset1::Constant>(
            element::i32,
            Shape({ newReshapeConstValues.size() }),
            newReshapeConstValues);

        const std::shared_ptr<Node> resultConstant = fold<ov::opset1::Reshape>(
            broadcastedConstant,
            newReshapeConstant,
            reshape->get_special_zero());

        replace_node(originalConstant, resultConstant);
    };

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(reshape, defaultPrecisions, 0);

    if (dequantization.subtract != nullptr) {
        replaceConstant(reshape, dequantization.subtractConstant);
    }

    if (dequantization.multiply != nullptr) {
        replaceConstant(reshape, dequantization.multiplyConstant);
    }
}

} // namespace

bool ReshapeTransformation::transform(ov::pass::pattern::Matcher &m) {
    std::shared_ptr<ov::opset1::Reshape> reshape = ov::as_type_ptr<ov::opset1::Reshape>(m.get_match_root());
    if (NetworkHelper::isConstantPath(reshape)) {
        return false;
    }

    if (!canBeTransformed(reshape)) {
        return false;
    }

    reshape = ov::as_type_ptr<ov::opset1::Reshape>(NetworkHelper::separateInStandaloneBranch(reshape, defaultPrecisions));
    reshapeDequantizationConstant(reshape, defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(reshape, NetworkHelper::getDequantization(reshape, defaultPrecisions, 0));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool ReshapeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

inline size_t getLastNotBroadcastedDimension(const Shape& shape) {
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] != 1ul) {
            return i;
        }
    }
    return 0;
}

inline size_t getFirstChangedDimension(const PartialShape& shape1, const PartialShape& shape2) {
    const size_t minSize = std::min(shape1.rank().get_length(), shape2.rank().get_length());
    size_t i = 0;
    for (; i < minSize; ++i) {
        if (shape1[i] != shape2[i]) {
            return i;
        }
    }
    return i;
}

bool ReshapeTransformation::canBeTransformed(const std::shared_ptr<Node>& op) const {
    if (!LayerTransformation::canBeTransformed(op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    bool ignorePerTensorQuantizationCheck = false;
    if (reshapeIgnorePerTensorQuantizationCheck) {
        const auto inputs = op->get_output_target_inputs(0);
        if (inputs.size() == 1ul) {
            const auto consumer = inputs.begin()->get_node();
            ignorePerTensorQuantizationCheck = ov::as_type<ov::opset1::MatMul>(consumer) != nullptr;
        }
    }

    if (!ignorePerTensorQuantizationCheck &&
        (((dequantization.subtract == nullptr) || NetworkHelper::isScalarLike(dequantization.subtractConstant)) &&
        ((dequantization.multiply == nullptr) || NetworkHelper::isScalarLike(dequantization.multiplyConstant)))) {
        return true;
    }

    const PartialShape outputPShape = op->get_output_partial_shape(0);
    if (outputPShape.size() < 2 || outputPShape[1].is_dynamic()) {
        return false;
    }

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

    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiplyConstant->get_shape();
    Shape multiplyShapeWithBatch = multiplyShape;
    if ((dequantization.multiply != nullptr) &&
        (multiplyShapeWithBatch.size() > 1ul) &&
        (multiplyShapeWithBatch.size() < inputRank)) {
        multiplyShapeWithBatch.insert(multiplyShapeWithBatch.begin(), 1ul);
    }

    const size_t outputChannel = static_cast<size_t>(outputPShape[1].get_length());
    if ((subtractShapeWithBatch.size() > 1) && (outputChannel < subtractShapeWithBatch[1])) {
        return false;
    }
    if ((multiplyShapeWithBatch.size() > 1) && (outputChannel < multiplyShapeWithBatch[1])) {
        return false;
    }

    if (outputPShape.is_static() &&
        (((subtractShapeWithBatch.size() > 1) && ((outputChannel % subtractShapeWithBatch[1]) != 0)) ||
         ((multiplyShapeWithBatch.size() > 1) && (outputChannel % multiplyShapeWithBatch[1] != 0)))) {
        return false;
    }

    return canBeTransformed(subtractShapeWithBatch, multiplyShapeWithBatch, inputPShape, outputPShape);
}

bool ReshapeTransformation::canBeTransformed(
    const ov::Shape& subtractShape,
    const ov::Shape& multiplyShape,
    const ov::PartialShape& inputShape,
    const ov::PartialShape& outputShape) {
    const size_t inputRank = inputShape.rank().get_length();
    const size_t outputRank = outputShape.rank().get_length();

    if ((inputRank < 2ul) || (outputRank < 2ul) || (inputShape[0] != outputShape[0])) {
        return false;
    }

    const size_t lastNotBroadcastedDimension = std::max(getLastNotBroadcastedDimension(subtractShape), getLastNotBroadcastedDimension(multiplyShape));
    const size_t firstChangedDimension = getFirstChangedDimension(inputShape, outputShape);
    // LPT supports channel on the second dimension natively <= reshape transformation supports more shapes for this case
    if ((lastNotBroadcastedDimension == 1ul) && (firstChangedDimension == 1ul)) {
        return true;
    }

    if (lastNotBroadcastedDimension >= firstChangedDimension) {
        return false;
    }

    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
