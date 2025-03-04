// Copyright (C) 2018-2025 Intel Corporation
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

        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ConcatTransformation::transform(ov::pass::pattern::Matcher &m) {
    if (!canBeTransformed(m.get_match_root())) {
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

    const auto& concat_out_shape = concat->get_output_partial_shape(0);
    const auto axis = ov::util::try_normalize_axis(concat->get_axis(), concat_out_shape.rank(), *concat);
    const bool scalar_equal_constants_requested = concat_out_shape[axis].is_dynamic();

    auto adaptConstForConcatenation = [scalar_equal_constants_requested](
                                          const std::shared_ptr<opset1::Constant>& constant,
                                          const Shape& targetShape) {
        if (scalar_equal_constants_requested) {
            OPENVINO_ASSERT(targetShape.empty(), "scalar_equal_constants_requested implies targetShape is empty");
            return std::make_shared<opset1::Constant>(*constant, ov::Shape{});
        } else {
            auto targetShapeConst = std::make_shared<opset1::Constant>(element::i64, Shape{ targetShape.size() }, targetShape);
            auto bcastedConst = ov::as_type_ptr<opset1::Constant>(fold<ov::opset1::Broadcast>(constant, targetShapeConst));
            OPENVINO_ASSERT(bcastedConst, "adaptConstForConcatenation must return constant");
            return bcastedConst;
        }
    };

    const bool someDqInLowPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return value.isLowPrecision(); });

    const bool someDqInFpPrecision = std::any_of(
        layerDequantizations.begin(),
        layerDequantizations.end(),
        [](const FakeQuantizeDequantization& value) { return !value.isLowPrecision(); });

    const bool DqWithDifferentPrecision = someDqInLowPrecision && someDqInFpPrecision;

    OutputVector dataNodes;
    NodeVector convertNodes;

    using ConstVector = std::vector<std::shared_ptr<opset1::Constant>>;
    ConstVector subConstants;
    ConstVector mulConstants;
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

        const auto targetShape = [&]() {
            if (scalar_equal_constants_requested)
                return ov::Shape{};
            Shape targetShape(concat->get_input_partial_shape(i).rank().get_length(), 1ul);
            targetShape[axis] = concat->get_input_partial_shape(i)[axis].get_length();
            return targetShape;
        }();

        if (!allDequantizationShiftAreZero) {
            auto subtractInput = dequantization.subtract == nullptr ?
                    std::make_shared<ov::opset1::Constant>(
                        (allDequantizationShiftConvertAreNotZero ?
                            PrecisionBeforeConvert :
                            deqPrecision),
                        targetShape,
                        std::vector<float>({ 0.f })) :
                adaptConstForConcatenation(dequantization.subtractConstant, targetShape);
            if (allDequantizationShiftConvertAreNotZero) {
                if (subtractConvert == nullptr && dequantization.subtractConvert != nullptr) {
                    subtractConvert = dequantization.subtractConvert;
                }
            } else if (dequantization.subtractConvert != nullptr) {
                const auto& dstType = dequantization.subtractConvert->get_convert_element_type();
                subtractInput = ov::as_type_ptr<opset1::Constant>(foldConvert(subtractInput, dstType));
                OPENVINO_ASSERT(subtractInput, "foldConvert must finish successfully for the concatenated subtract constant");
                NetworkHelper::copyInfo(dequantization.subtractConvert, subtractInput);
            }
            subConstants.push_back(subtractInput);
        }

        if (!allDequantizationMultiplyAreZero) {
            mulConstants.push_back(dequantization.multiply == nullptr ?
                std::make_shared<ov::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 1.0f })) :
                adaptConstForConcatenation(dequantization.multiplyConstant, targetShape));
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

    auto concat_constants_if_needed = [&](const ConstVector& constants) -> std::shared_ptr<ov::Node> {
        OPENVINO_ASSERT(!constants.empty(), "concat_constants_if_needed expects non empty constants vec");
        if (constants.size() == 1ul) {
            return constants[0];
        }
        if (scalar_equal_constants_requested) {
            if (ov::shape_size(constants[0]->get_shape()) == 1) {
                const auto ref_value = constants[0]->cast_vector<float>();
                if (std::all_of(constants.cbegin() + 1, constants.cend(), [&ref_value](const auto& constant) {
                        return constant->template cast_vector<float>() == ref_value;
                    })) {
                    return constants[0];
                }
            }
            OPENVINO_THROW("in case of dynamic concatenation dim all constants must be scalar and equal");
        }
        ov::OutputVector concatInputs;
        std::transform(constants.begin(), constants.end(), std::back_inserter(concatInputs), [](const auto& constant) {
            return constant->output(0);
        });
        return fold<ov::opset1::Concat>(concatInputs, axis);
    };

    if (!subConstants.empty()) {
        auto subtractNode = concat_constants_if_needed(subConstants);
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
                NetworkHelper::toScalarIfPossible(concat_constants_if_needed(mulConstants))),
            layerDequantizations[0].multiply->get_output_element_type(0));

        NetworkHelper::copyInfo({ concat, multiply }, multiply);
        multiply->set_friendly_name(concat->get_friendly_name() + "/DequantizationMultyply");
        lastDequantization = multiply;
    }

    NetworkHelper::insertDequantizationAfter(concat, lastDequantization, newConcat);
    NetworkHelper::copyInfo(concat, newConcat);
    updateOutput(lastDequantization, newConcat);

    OPENVINO_DEBUG("LPT: done: ", newConcat);
    return true;
}

bool ConcatTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

bool ConcatTransformation::canBeTransformed(const std::shared_ptr<Node>& layer) const {
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

    auto base_dq_check = [&](const FakeQuantizeDequantization& dequantization) {
        return !dequantization.empty() && (!updatePrecisions || dequantization.isLowPrecision());
    };

    const size_t normalizedAxis = ov::util::try_normalize_axis(axis, outRank, *concat);
    if (outPShape[normalizedAxis].is_dynamic()) {
        // in case of dynamic dimension we can propagate all dequantizations only if they are all scalar and equal,
        // since DQ broadcast is impossible (requested shape is unknown), and only single scalar DQ after Concat can be set
        const auto dequantization_ref = NetworkHelper::getDequantization(concat, defaultPrecisions, 0);
        if (!base_dq_check(dequantization_ref) || !dequantization_ref.isPerTensor())
            return false;

        auto extract_values = [](const std::shared_ptr<ov::op::v0::Constant>& constant) {
            return constant ? constant->cast_vector<float>() : std::vector<float>();
        };
        const auto ref_shifts = extract_values(dequantization_ref.subtractConstant);
        const auto ref_scales = extract_values(dequantization_ref.multiplyConstant);

        for (size_t i = 1ul; i < concat->get_input_size(); i++) {
            const auto cur_dequantization = NetworkHelper::getDequantization(concat, defaultPrecisions, i);
            if (!base_dq_check(dequantization_ref) ||
                ref_shifts != extract_values(cur_dequantization.subtractConstant) ||
                ref_scales != extract_values(cur_dequantization.multiplyConstant))
                return false;
        }
        return true;
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
        const std::shared_ptr<Node>& constant,
        ov::element::Type& const_precision) {
        if (constant == nullptr) {
            return true;
        }
        if (const_precision == element::dynamic) {
            const_precision = constant->get_element_type();
            return true;
        }
        return const_precision == constant->get_element_type();
    };

    element::Type precision;
    element::Type const_precision;

    for (size_t i = 0ul; i < concat->get_input_size(); i++) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(concat, defaultPrecisions, i);
        if (!base_dq_check(dequantization))
            return false;

        if (((dequantization.subtract != nullptr) && (!checkConstShape(dequantization.subtractConstant))) ||
            ((dequantization.multiply != nullptr) && (!checkConstShape(dequantization.multiplyConstant)))) {
            return false;
        }

        if (precision == element::dynamic) {
            precision = dequantization.data.get_element_type();
        } else if (precision != dequantization.data.get_element_type()) {
            return false;
        }

        if (!check_const_precision(dequantization.subtractConvert, const_precision) ||
            ((dequantization.subtractConvert == nullptr) && !check_const_precision(dequantization.subtractConstant, const_precision)) ||
            !check_const_precision(dequantization.multiplyConstant, const_precision)) {
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
