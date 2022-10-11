// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/pad.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

PadTransformation::PadTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(PadTransformation);
    auto mul = pattern::wrap_type<opset1::Multiply>();
    auto padsBegin = pattern::wrap_type<opset1::Constant>();
    auto padsEnd = pattern::wrap_type<opset1::Constant>();
    auto padsValue = pattern::wrap_type<opset1::Constant>();
    auto matcher = pattern::wrap_type<opset1::Pad>({ mul, padsBegin, padsEnd, padsValue });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool PadTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto pad = ov::as_type_ptr<opset1::Pad>(NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions));
    const auto padConstant = ov::as_type_ptr<opset1::Constant>(pad->get_input_node_shared_ptr(3));
    const auto padConstantValue = padConstant->cast_vector<float>()[0];

    const auto padsBegin = pad->get_pads_begin();
    const auto padsEnd = pad->get_pads_end();
    const auto padMode = pad->get_pad_mode();

    auto dequantization = NetworkHelper::getDequantization(pad, defaultPrecisions);

    if (padMode == op::PadMode::CONSTANT) {
        auto bcastConstant = [&](const std::shared_ptr<opset1::Constant> &constant) {
            size_t padIdx = 0;
            for (size_t i = 0; i < padsBegin.size(); ++i) {
                if (padsBegin[i] != 0 || padsEnd[i] != 0) {
                    padIdx = i;
                    break;
                }
            }

            const auto inputPShape = pad->get_input_partial_shape(0);
            assert(inputPShape[padIdx].is_static());
            assert(inputPShape.rank().is_static());
            auto bcastedShape = Shape(inputPShape.rank().get_length(), 1ul);
            bcastedShape[padIdx] = inputPShape[padIdx].get_length();

            const auto bCastConst = opset1::Constant::create(element::i32, Shape{bcastedShape.size()}, bcastedShape);
            return ov::as_type_ptr<opset1::Constant>(fold<opset1::Broadcast>(constant, bCastConst));
        };

        if (dequantization.subtract && shape_size(dequantization.subtractConstant->get_shape()) == 1ul) {
            const auto broadcastedConstant = bcastConstant(dequantization.subtractConstant);
            replace_node(dequantization.subtractConstant, broadcastedConstant);
            dequantization.subtractConstant = broadcastedConstant;
        }

        if (padConstantValue != 0.f && shape_size(dequantization.multiplyConstant->get_shape()) == 1ul) {
            const auto broadcastedConstant = bcastConstant(dequantization.multiplyConstant);
            replace_node(dequantization.multiplyConstant, broadcastedConstant);
            dequantization.multiplyConstant = broadcastedConstant;
        }
    }

    auto foldConstantIfNecessary = [&padMode, &padsBegin, &padsEnd](
        const std::shared_ptr<opset1::Constant>& constant,
        const std::shared_ptr<opset1::Pad>& pad,
        float padVal) {
        const auto constantShape = constant->get_shape();
        if (shape_size(constantShape) == 1ul) {
            return NetworkHelper::toScalar(constant);
        }

        std::vector<size_t> padsForConstantBegin(constantShape.size(), 0ul);
        std::vector<size_t> padsForConstantEnd(constantShape.size(), 0ul);
        bool foldingIsNecessary = false;

        // folding is necessary when dequantization and padding by the same dimension
        for (size_t i = 0; i < constantShape.size(); ++i) {
            if (padsBegin[i] != 0ul && constantShape[i] != 1ul) {
                foldingIsNecessary = true;
                padsForConstantBegin[i] = padsBegin[i];
            }

            if (padsEnd[i] != 0ul && constantShape[i] != 1ul) {
                foldingIsNecessary = true;
                padsForConstantEnd[i] = padsEnd[i];
            }
        }

        if (foldingIsNecessary) {
            const auto beginConst = opset1::Constant::create(element::u32, { padsForConstantBegin.size() }, padsForConstantBegin);
            const auto endConst = opset1::Constant::create(element::u32, { padsForConstantEnd.size() }, padsForConstantEnd);
            const auto padValueConstant = opset1::Constant::create(constant->get_element_type(), Shape{}, { padVal });
            const auto foldedConstant = fold<opset1::Pad>(constant, beginConst, endConst, padValueConstant, padMode);
            return ov::as_type_ptr<opset1::Constant>(foldedConstant);
        } else {
            return constant;
        }
    };

    if (dequantization.subtract) {
        const auto normalizedSubConst = NetworkHelper::normalizeDequantizationShape(dequantization.subtract);
        float padValueForSub = padConstantValue;
        if (padMode == op::PadMode::CONSTANT) {
            padValueForSub = 0.f;
        }

        const auto newSubConstant = foldConstantIfNecessary(normalizedSubConst, pad, padValueForSub);
        replace_node(normalizedSubConst, newSubConstant);
        dequantization.subtractConstant = newSubConstant;
    }

    {
        const auto normalizedMulConst = NetworkHelper::normalizeDequantizationShape(dequantization.multiply);
        float padValueForMul = padConstantValue;
        if (padMode == op::PadMode::CONSTANT) {
            padValueForMul = 1.f;
        }

        const auto newMulConstant = foldConstantIfNecessary(normalizedMulConst, pad, padValueForMul);
        replace_node(normalizedMulConst, newMulConstant);
        dequantization.multiplyConstant = newMulConstant;
    }

    // we must convert pad value in low precision
    const auto convertedZero = opset1::Constant::create(dequantization.data.get_element_type(), Shape{}, { padConstantValue });
    pad->set_argument(3, convertedZero);

    moveDequantizationAfter(context, pad, dequantization, true);
    return true;
}

bool PadTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformedSpatialDimension(context, op)) {
        return false;
    }

    const auto pad = ov::as_type_ptr<opset1::Pad>(op);
    if (!pad) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    const auto mode = pad->get_pad_mode();
    if (mode == op::PadMode::CONSTANT) {
        auto padAndDqByTheSameDimension = [&](const std::shared_ptr<opset1::Constant>& deqConst) {
            const auto padsBegin = pad->get_pads_begin();
            const auto padsEnd = pad->get_pads_end();

            int beginNonZeroIdx = -1;
            for (size_t i = 0; i < padsBegin.size(); ++i) {
                const bool padDimensionNotUnique = (beginNonZeroIdx != -1) && (padsBegin[i] != 0);
                if (padDimensionNotUnique) {
                    return false;
                }

                if (padsBegin[i] != 0) {
                    beginNonZeroIdx = static_cast<int>(i);
                }
            }

            int endNonZeroIdx = -1;
            for (size_t i = 0; i < padsEnd.size(); ++i) {
                const bool padDimensionNotUnique = (endNonZeroIdx != -1) && (padsEnd[i] != 0);
                if (padDimensionNotUnique) {
                    return false;
                }

                if (padsEnd[i] != 0) {
                    endNonZeroIdx = static_cast<int>(i);
                }
            }

            if ((beginNonZeroIdx != endNonZeroIdx) && (beginNonZeroIdx != -1) && (endNonZeroIdx != -1)) {
                return false;
            }

            const size_t paddingDimension = beginNonZeroIdx != -1 ? beginNonZeroIdx : endNonZeroIdx;
            const auto padInputPShape = pad->get_input_partial_shape(0);
            const auto padInputRank = padInputPShape.rank();
            if (padInputRank.is_dynamic() || padInputPShape[paddingDimension].is_dynamic()) {
                return false;
            }


            const size_t inputRankValue = padInputRank.get_length();
            auto deqShape = deqConst->get_shape();
            if (shape_size(deqShape) > 1ul) {
                while (deqShape.size() < inputRankValue) {
                    deqShape.insert(deqShape.begin(), 1ul);
                }

                for (size_t i = 0; i < deqShape.size(); ++i) {
                    const bool deqAndPadDimensionsMismatched = (deqShape[i] > 1ul) && (i != paddingDimension);
                    if (deqAndPadDimensionsMismatched) {
                        return false;
                    }
                }
            }

            return true;
        };

        if (dequantization.subtract && !padAndDqByTheSameDimension(dequantization.subtractConstant)) {
            return false;
        }

        const auto constant = ov::as_type_ptr<opset1::Constant>(pad->get_input_node_shared_ptr(3));
        const auto constantValue = constant->cast_vector<float>()[0];
        if (constantValue != 0.f && !padAndDqByTheSameDimension(dequantization.multiplyConstant)) {
            return false;
        }
    }

    if (mode == op::PadMode::REFLECT) {
        auto deqShape = dequantization.multiplyConstant->get_shape();
        if (shape_size(deqShape) == 1ul) {
            return true;
        } else {
            const auto padInputRank = pad->get_input_partial_shape(0).rank();
            if (padInputRank.is_dynamic()) {
                return false;
            }

            const size_t inputRankValue = padInputRank.get_length();
            while (deqShape.size() < inputRankValue) {
                deqShape.insert(deqShape.begin(), 1ul);
            }

            const auto padsBegin = pad->get_pads_begin();
            const auto padsEnd = pad->get_pads_end();

            // PadTransformation with "REFLECT" mode doesn't support dequantization and padding by the same dimension
            for (size_t i = 0; i < deqShape.size(); ++i) {
                if (deqShape[i] != 1ul && (padsBegin[i] != 0ul || padsEnd[i] != 0ul)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool PadTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
