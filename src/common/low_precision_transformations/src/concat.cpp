// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/concat.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/fake_quantize_dequantization.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ConcatTransformation::ConcatTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(ConcatTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::Concat>();

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

bool ConcatTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<ngraph::opset1::Concat> concat = ngraph::as_type_ptr<ngraph::opset1::Concat>(m.get_match_root());
    if (!canBeTransformed(context, concat)) {
        return false;
    }

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
        auto broadcast = fold<ngraph::opset1::Broadcast>(operation, targetShapeConst);
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
    const auto axis = ngraph::normalize_axis(concat->get_friendly_name(),
        concat->get_axis(),
        concat->get_output_partial_shape(0).rank());

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
                    std::make_shared<ngraph::opset1::Constant>(
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
                std::make_shared<ngraph::opset1::Constant>(deqPrecision, targetShape, std::vector<float>({ 1.0f })) :
                broadcastElementWiseConst(dequantization.multiplyConstant, targetShape));
        }
    }

    const auto newConcat = concat->clone_with_new_inputs(dataNodes);

    std::shared_ptr<ngraph::Node> lastDequantization = newConcat;
    if (!convertNodes.empty()) {
        const auto convert = convertNodes[0]->clone_with_new_inputs({ newConcat });

        NetworkHelper::copyInfo({ concat, convert }, convert);
        convert->set_friendly_name(concat->get_friendly_name() + "/DequantizationConvert");
        lastDequantization = convert;
    }

    if (!subConstants.empty()) {
        std::shared_ptr<ov::Node> subtractNode = subConstants.size() == 1ul ?
            subConstants[0] :
            ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(subConstants, axis);
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
        const auto multiply = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
            opset1::Multiply(
                lastDequantization,
                NetworkHelper::toScalarIfPossible(mulConstants.size() == 1ul ?
                    mulConstants[0] :
                    ngraph::pass::low_precision::fold<ngraph::opset1::Concat>(mulConstants, axis))),
            layerDequantizations[0].multiply->get_output_element_type(0));

        NetworkHelper::copyInfo({ concat, multiply }, multiply);
        multiply->set_friendly_name(concat->get_friendly_name() + "/DequantizationMultyply");
        lastDequantization = multiply;
    }

    NetworkHelper::insertDequantizationAfter(concat, lastDequantization, newConcat);
    NetworkHelper::copyInfo(concat, newConcat);
    updateOutput(context, lastDequantization, newConcat);
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

    const size_t normalizedAxis = ngraph::normalize_axis(concat->get_friendly_name(), axis, outRank);
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

    element::Type precision;
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
    }
    return true;
}

void ConcatTransformation::fillDequantizationNodes(
    const std::vector<FakeQuantizeDequantization>& layerDequantizations,
    const std::shared_ptr<Node> layer,
    NodeVector& convertNodes,
    NodeVector& subtractNodes,
    NodeVector& multiplyNodes) const {
    if (layerDequantizations.size() > 1ul) {
        auto broadcastElementWiseConst = [](
            // FakeQuantize constant shape must be broadcastable to the shape on data.
            std::shared_ptr<ngraph::opset1::Constant> operation,
            const ngraph::Shape targetShape) -> std::shared_ptr<Node> {
                auto targetShapeConst = opset1::Constant::create(element::i64, ngraph::Shape{ targetShape.size() }, targetShape);
                auto broadcast = fold<ngraph::opset1::Broadcast>(operation, targetShapeConst);
                return broadcast;
        };

        bool allDequantizationShiftAreZero = true;
        bool allDequantizationMultiplyAreZero = true;
        for (const auto& dequantization : layerDequantizations) {
            if (dequantization.subtract != nullptr) {
                allDequantizationShiftAreZero = false;
            }
            if (dequantization.multiply != nullptr) {
                allDequantizationMultiplyAreZero = false;
            }
        }

        for (size_t i = 0; i < layerDequantizations.size(); ++i) {
            const auto& dequantization = layerDequantizations[i];
            const ngraph::element::Type precision = deqPrecision;
            ngraph::Shape targetShape(layer->get_input_partial_shape(i).rank().get_length(), 1ul);
            targetShape[1] = layer->get_input_partial_shape(i)[1].get_length();

            if (dequantization.convert != nullptr) {
                convertNodes.push_back(dequantization.convert);
            }

            if (!allDequantizationShiftAreZero) {
                subtractNodes.push_back(dequantization.subtract == nullptr ?
                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 0.f })) :
                    broadcastElementWiseConst(dequantization.subtractConstant, targetShape));
            }

            if (!allDequantizationMultiplyAreZero) {
                multiplyNodes.push_back(dequantization.multiply == nullptr ?
                    std::make_shared<ngraph::opset1::Constant>(precision, targetShape, std::vector<float>({ 1.0f })) :
                    broadcastElementWiseConst(dequantization.multiplyConstant, targetShape));
            }
        }
    } else {
        // TODO: check constant shapes here - has to be scalar
        if (layerDequantizations[0].convert != nullptr) {
            convertNodes.push_back(layerDequantizations[0].convert);
        }

        if (layerDequantizations[0].subtract != nullptr) {
            subtractNodes.push_back(layerDequantizations[0].subtract->input_value(1).get_node_shared_ptr());
        }

        if (layerDequantizations[0].multiply != nullptr) {
            multiplyNodes.push_back(layerDequantizations[0].multiply->input_value(1).get_node_shared_ptr());
        }
    }
}

bool ConcatTransformation::isHandled(const TransformationContext& context, const std::vector<std::shared_ptr<ngraph::Node>>& quantizationOperations) {
    for (const std::shared_ptr<ngraph::Node>& quantizationLayer : quantizationOperations) {
        if (context.quantizedFakeQuantizeNames.find(quantizationLayer->get_friendly_name()) != context.quantizedFakeQuantizeNames.end()) {
            return true;
        }
    }

    return false;
}

bool ConcatTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer) {
    const auto concat = as_type_ptr<const opset1::Concat>(layer);
    if (concat == nullptr)
        return false;
    return concat->get_output_partial_shape(0).rank().is_static();
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
