// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/util/log.hpp"

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ConcatTransformation::ConcatTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ConcatTransformation);
    auto matcher = ov::pass::pattern::wrap_type<opset1::Concat>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ConcatTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto concat = ov::as_type_ptr<ov::opset1::Concat>(NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions));
    std::vector<FakeQuantizeDequantization> layerDequantizations;
    layerDequantizations.reserve(concat->get_input_size());
    for (size_t parentIndex = 0ul; parentIndex < concat->get_input_size(); parentIndex++) {
        FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, defaultPrecisions, parentIndex);
        if (dequantization.empty()) {
            return false;
        }
        layerDequantizations.push_back(dequantization);
    }

    bool allDequantizationShiftAreZero = true;
    bool allDequantizationShiftConvertAreNotZero = true;
    bool allDequantizationMultiplyAreZero = true;
    element::Type PrecisionBeforeConvert;
    for (const auto& dequantization : layerDequantizations) {
        if (dequantization.subtract != nullptr) {
            allDequantizationShiftAreZero = false;
            if (dequantization.subtractConvert == nullptr) {
                allDequantizationShiftConvertAreNotZero = false;
            } else {
                PrecisionBeforeConvert = dequantization.subtractConstant->get_element_type();
            }
        }

        if (dequantization.multiply != nullptr) {
            allDequantizationMultiplyAreZero = false;
        }

        if (!allDequantizationShiftAreZero && !allDequantizationMultiplyAreZero &&
            !allDequantizationShiftConvertAreNotZero) {
            break;
        }
    }
    if (allDequantizationShiftAreZero) {
        allDequantizationShiftConvertAreNotZero = false;
    }

    // constant shape must be broadcastable to the shape on data.
    auto broadcastElementWiseConst = [](std::shared_ptr<opset1::Constant> operation, const Shape targetShape) {
        auto targetShapeConst = std::make_shared<opset1::Constant>(element::i64, Shape{ targetShape.size() }, targetShape);
        auto broadcast = fold<ov::opset1::Broadcast>(operation, targetShapeConst);
        return broadcast;
    };

    bool someDqInLowPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return value.isLowPrecision(); });

    bool someDqInFpPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return !value.isLowPrecision(); });

    bool DqWithDifferentPrecision = someDqInLowPrecision && someDqInFpPrecision;
    const auto axis =
        ov::util::try_normalize_axis(concat->get_axis(), concat->get_output_partial_shape(0).rank(), *concat);

    OutputVector dataNodes;
    NodeVector convertNodes;
    NodeVector subConstants;
    NodeVector mulConstants;
    std::shared_ptr<opset1::Convert> subtractConvert = nullptr;
    for (size_t i = 0; i < layerDequantizations.size(); ++i) {
        const auto& dequantization = layerDequantizations[i];

        if (DqWithDifferentPrecision && dequantization.isLowPrecision()) {
            dataNodes.push_back(dequantization.convert);
        } else {
            dataNodes.push_back(dequantization.data);
        }

        if (dequantization.convert != nullptr) {
            convertNodes.push_back(dequantization.convert);
        }

        Shape targetShape(concat->get_input_partial_shape(i).rank().get_length(), 1ul);
        targetShape[axis] = concat->get_input_partial_shape(i)[axis].get_length();

        if (!allDequantizationShiftAreZero) {
            auto subtractInput = dequantization.subtract == nullptr ?
                    std::make_shared<ov::opset1::Constant>(
                        (allDequantizationShiftConvertAreNotZero ?
                            PrecisionBeforeConvert :
                            deqPrecision),
                        targetShape,
                        std::vector<float>({ 0.f })) :
                broadcastElementWiseConst(dequantization.subtractConstant, targetShape);
            if (allDequantizationShiftConvertAreNotZero) {
                if (subtractConvert == nullptr && dequantization.subtractConvert != nullptr) {
                    subtractConvert = dequantization.subtractConvert;
                }
            } else if (dequantization.subtractConvert != nullptr) {
                subtractInput = foldConvert(subtractInput, dequantization.subtractConvert->get_convert_element_type());
                NetworkHelper::copyInfo(dequantization.subtractConvert, subtractInput);
            }
            subConstants.push_back(subtractInput);
        }

        if (!allDequantizationMultiplyAreZero) {
            mulConstants.push_back(dequantization.multiply == nullptr ?
                std::make_shared<ov::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 1.0f })) :
                broadcastElementWiseConst(dequantization.multiplyConstant, targetShape));
        }
    }

    const auto newConcat = concat->clone_with_new_inputs(dataNodes);

    std::shared_ptr<ov::Node> lastDequantization = newConcat;
    if (!convertNodes.empty()) {
        const auto convert = convertNodes[0]->clone_with_new_inputs({ newConcat });

        NetworkHelper::copyInfo({ concat, convert }, convert);
        convert->set_friendly_name(concat->get_friendly_name() + "/DequantizationConvert");
        lastDequantization = convert;
    }

    if (!subConstants.empty()) {
        std::shared_ptr<ov::Node> subtractNode = subConstants.size() == 1ul ?
            subConstants[0] :
            ov::pass::low_precision::fold<ov::opset1::Concat>(subConstants, axis);
        if (subtractConvert != nullptr)
            subtractNode = subtractConvert->clone_with_new_inputs({subtractNode});
        const auto subtract = std::make_shared<opset1::Subtract>(
            lastDequantization,
            NetworkHelper::toScalarIfPossible(subtractNode));

        NetworkHelper::copyInfo({ concat, subtract }, subtract);
        subtract->set_friendly_name(concat->get_friendly_name() + "/DequantizationSubtract");
        lastDequantization = subtract;
    }

    if (!mulConstants.empty()) {
        const auto multiply = std::make_shared<ov::op::TypeRelaxed<opset1::Multiply>>(
            opset1::Multiply(
                lastDequantization,
                NetworkHelper::toScalarIfPossible(mulConstants.size() == 1ul ?
                    mulConstants[0] :
                    ov::pass::low_precision::fold<ov::opset1::Concat>(mulConstants, axis))),
            layerDequantizations[0].multiply->get_output_element_type(0));

        NetworkHelper::copyInfo({ concat, multiply }, multiply);
        multiply->set_friendly_name(concat->get_friendly_name() + "/DequantizationMultyply");
        lastDequantization = multiply;
    }

    NetworkHelper::insertDequantizationAfter(concat, lastDequantization, newConcat);
    NetworkHelper::copyInfo(concat, newConcat);
    updateOutput(context, lastDequantization, newConcat);

    OPENVINO_DEBUG("LPT: done: ", newConcat);
    return true;
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

bool ConcatTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    std::shared_ptr<opset1::Concat> concat = ov::as_type_ptr<opset1::Concat>(layer);
    if (concat == nullptr) {
        return false;
    }

    const auto& axis = concat->get_axis();
    const auto& outPShape = concat->get_output_partial_shape(0);
    const auto& outRank = outPShape.rank();
    if (outRank.is_dynamic()) {
        return false;
    }

    const size_t normalizedAxis = ov::util::try_normalize_axis(axis, outRank, *concat);
    if (outPShape[normalizedAxis].is_dynamic()) {
        return false;
    }

    auto checkConstShape = [&normalizedAxis, &outRank](const std::shared_ptr<opset1::Constant>& constant) {
        const size_t rankValue = outRank.get_length();
        Shape constantShape = constant->get_shape();

        while (constantShape.size() < rankValue) {
            constantShape.insert(constantShape.begin(), 1ul);
        }

        const auto dqDimensionsCount = std::count_if(constantShape.begin(), constantShape.end(), [](size_t elem) { return elem > 1; });
        const bool dqOnlyByConcatAxis = (dqDimensionsCount == 0) || (dqDimensionsCount == 1 && constantShape[normalizedAxis] != 1ul);
        return dqOnlyByConcatAxis;
    };

    const auto check_const_precision = [](
        const FakeQuantizeDequantization& dequantization,
        const std::shared_ptr<Node>& constant,
        ov::element::Type& const_precision) {
        if (constant == nullptr) {
            return true;
        }
        if (const_precision == element::undefined) {
            const_precision = constant->get_element_type();
            return true;
        }
        return const_precision == constant->get_element_type();
    };

    element::Type precision;
    element::Type const_precision;

    for (size_t i = 0ul; i < concat->get_input_size(); i++) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, defaultPrecisions, i);
        if (dequantization.empty() || (updatePrecisions && !dequantization.isLowPrecision())) {
            return false;
        }

        if (((dequantization.subtract != nullptr) && (!checkConstShape(dequantization.subtractConstant))) ||
            ((dequantization.multiply != nullptr) && (!checkConstShape(dequantization.multiplyConstant)))) {
            return false;
        }

        if (precision == element::undefined) {
            precision = dequantization.data.get_element_type();
        } else if (precision != dequantization.data.get_element_type()) {
            return false;
        }

        if (!check_const_precision(dequantization, dequantization.subtractConvert, const_precision) ||
            ((dequantization.subtractConvert == nullptr) && !check_const_precision(dequantization, dequantization.subtractConstant, const_precision)) ||
            !check_const_precision(dequantization, dequantization.multiplyConstant, const_precision)) {
            return false;
        }
    }
    return true;
}

bool ConcatTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer) {
    const auto concat = as_type_ptr<const opset1::Concat>(layer);
    if (concat == nullptr)
        return false;
    return concat->get_output_partial_shape(0).rank().is_static();
}

} // namespace low_precision
} // namespace pass
} // namespace ov
