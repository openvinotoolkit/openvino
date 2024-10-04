// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/layer_transformation.hpp"
#include "low_precision/network_helper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>
#include <queue>
#include "itt.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace pass {
namespace low_precision {

const std::vector<element::Type>& precision_set::get_int8_support() {
    static const std::vector<element::Type> int8_support = {
        ov::element::u8,  ov::element::i8
    };
    return int8_support;
}

const std::vector<element::Type>& precision_set::get_int8_int16_int32_support() {
    static const std::vector<element::Type> int8_int16_int32_support = {
        ov::element::u8,  ov::element::i8,
        ov::element::u16, ov::element::i16,
        ov::element::u32, ov::element::i32
    };
    return int8_int16_int32_support;
}

constexpr char LayerTransformation::originalLayerPostfix[];

LayerTransformation::LayerTransformation(const Params& params) :
    updatePrecisions(params.updatePrecisions),
    deqPrecision(params.deqPrecision),
    defaultPrecisions(params.defaultPrecisions),
    reshapeIgnorePerTensorQuantizationCheck(params.reshapeIgnorePerTensorQuantizationCheck),
    context(nullptr) {}

void LayerTransformation::setContext(TransformationContext* context) noexcept {
    this->context = context;
}

void LayerTransformation::setUpdatePrecisions(const bool updatePrecisions) {
    this->updatePrecisions = updatePrecisions;
}

void LayerTransformation::setDefaultPrecisions(const std::vector<ov::element::Type>& defaultPrecisions) {
    this->defaultPrecisions = defaultPrecisions;
}

bool LayerTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!isQuantized(layer, defaultPrecisions)) {
        return false;
    }

    return canBeTransformedStatic(layer);
}

bool LayerTransformation::canBeTransformedStatic(const std::shared_ptr<Node>& layer,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    const auto outputs = layer->outputs();
    if (std::any_of(outputs.begin(), outputs.end(),
        [](const Output<Node>& out) { return out.get_partial_shape().rank().is_dynamic(); })) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
    if (!dequantization.empty()) {
        auto perChannelQuantization = [](const PartialShape dataPShape, Shape constShape, size_t idxChannelDim) {
            if (ov::shape_size(constShape) == 1ul) {
                return true;
            }

            const auto rank = dataPShape.rank();
            if (rank.is_dynamic()) {
                return false;
            }

            if ((dataPShape.size() - constShape.size()) == 1ul) {
                constShape.insert(constShape.begin(), 1ul);
            }

            // special case: 1D const is assumed to imply per-channel
            if (constShape.size() == 1)
                return true;

            if ((constShape.size() >= 2ul) && (constShape[0] != 1ul)) {
                return false;
            }

            for (size_t i = 0; i < constShape.size(); ++i) {
                if ((constShape[i] != 1ul) && (i != idxChannelDim)) {
                    return false;
                }
            }
            return true;
        };

        if ((dequantization.subtract != nullptr) && (!perChannelQuantization(
            dequantization.subtract->get_output_partial_shape(0),
            dequantization.subtractConstant->get_shape(),
            dequantization.channelDimIndex))) {
            return false;
        }

        if ((dequantization.multiply != nullptr) && (!perChannelQuantization(
            dequantization.multiply->get_output_partial_shape(0),
            dequantization.multiplyConstant->get_shape(),
            dequantization.channelDimIndex))) {
            return false;
        }
    }

    return true;
}

bool LayerTransformation::canBeTransformedSpatialDimension(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!isQuantized(layer, defaultPrecisions)) {
        OPENVINO_DEBUG("LPT: early exit: not quantized");
        return false;
    }
    const auto outputs = layer->outputs();
    if (std::any_of(outputs.begin(), outputs.end(),
        [](const Output<Node>& out) { return out.get_partial_shape().rank().is_dynamic(); })) {
        OPENVINO_DEBUG("LPT: early exit: rank is dynamic");
        return false;
    }
    return true;
}

bool LayerTransformation::canSubtractBeHandled(const std::shared_ptr<Node>& op, const FakeQuantizeDequantization& dequantization) const {
    if (dequantization.empty() || (dequantization.subtract == nullptr)) {
        return true;
    }

    if (!updatePrecisions) {
        return true;
    }

    const element::Type operationType = dequantization.convert == nullptr ?
        dequantization.subtract->input(0).get_element_type() :
        dequantization.convert->input(0).get_element_type();

    if ((operationType != element::i8) && (operationType != element::u8)) {
        return false;
    }

    const auto parent = dequantization.subtract->input_value(1).get_node_shared_ptr();

    if (ov::is_type<ov::opset1::Constant>(parent)) {
        return true;
    } else if (ov::is_type<ov::opset1::Convert>(parent) && ov::is_type<ov::opset1::Constant>(parent->get_input_node_shared_ptr(0))) {
        const auto constant = parent->get_input_node_shared_ptr(0);
        const auto constantType = constant->output(0).get_element_type();
        return operationType == constantType;
    } else {
        return false;
    }
}

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
std::stringstream toStream(const std::vector<float>& dequantizationValues) {
    std::stringstream ss;
    const size_t scalesCount = dequantizationValues.size() > 9ul ? 9ul : dequantizationValues.size();
    ss << "{";
    for (size_t i = 0ul; i < scalesCount; ++i) {
        ss << dequantizationValues[i] << (i < (scalesCount - 1) ? "," : "");
    }
    ss << "}";
    return ss;
}

void LayerTransformation::printDequantizationInfo(const std::shared_ptr<Node>& layer) {
    auto fq = as_type_ptr<opset1::FakeQuantize>(layer);
    if (fq) {
        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(ov::as_type_ptr<opset1::FakeQuantize>(layer));
        std::cout <<
            layer->get_type_name() << (NetworkHelper::isConstantPath(layer) ? " on weights " : " on activations ") <<
            layer->get_friendly_name() << ":" << std::endl <<
            "   details  : " << quantizationDetails << std::endl;
    }
}

void LayerTransformation::printDequantizationInfo(const DataPrecision& dataPrecision) {
    std::cout << "   precision: " << dataPrecision << std::endl;
}

void LayerTransformation::printDequantizationValues(
    const std::vector<float>& dequantizationScales,
    const std::vector<float>& dequantizationShifts) {
    std::cout <<
        "   scales   : " << toStream(dequantizationScales).str() << std::endl <<
        "   shifts   : " << toStream(dequantizationShifts).str() << std::endl;
}
#endif

LayerTransformation::PrecisionDetails LayerTransformation::getPrecisionDetails(
    const size_t quantizationLevels,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues) {
    // TODO: workaround: hardcoded values
    const float zeroThreshold = 1.e-6f;
    const float quantizationIntervalAsymmetryThreshold = 0.002f;

    float asymmetricIntervalSideRatio = -static_cast<float>(quantizationLevels) / (quantizationLevels - 2.f);
    bool hasNegative = false;
    bool signedPrecision = true;
    bool unsignedPrecision = true;

    bool hasZeroPoint = false;
    bool thereIsAtLeastOneNormalValue = false;

    std::vector<size_t> fullRangeLevels = { levels::int4, levels::int8, levels::int16, levels::int32 };

    for (size_t i = 0; i < outputLowValues.size(); ++i) {
        if ((std::fabs(outputLowValues[i]) < zeroThreshold) && (std::fabs(outputHighValues[i]) < zeroThreshold)) {
            // both values are too small to identify preferable precision
            continue;
        }

        thereIsAtLeastOneNormalValue = true;

        const bool signedInterval = std::signbit(outputLowValues[i]) != std::signbit(outputHighValues[i]);
        const bool outputLowValueIsNotZero = std::fabs(outputLowValues[i]) >= zeroThreshold;
        if (signedInterval && outputLowValueIsNotZero) {
            // signed
            unsignedPrecision = false;
            hasNegative = true;

            if (outputHighValues[i] != 0.f) {
                auto it = std::find(fullRangeLevels.begin(), fullRangeLevels.end(), quantizationLevels);
                const float expectedRatio = it != fullRangeLevels.end() ? asymmetricIntervalSideRatio : -1.f;
                const float actualRatio = outputLowValues[i] / outputHighValues[i];
                const float actual = std::fabs((actualRatio - expectedRatio) / std::min(actualRatio, expectedRatio));
                if (actual > quantizationIntervalAsymmetryThreshold) {
                    hasZeroPoint = true;
                }
            } else {
                hasZeroPoint = true;
            }
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
            if (hasZeroPoint) {
                std::cout << "   actual: " << actual << ", threshold: " << quantizationIntervalAsymmetryThreshold << std::endl;
                std::cout << "   hasZeroPoint: " << (hasZeroPoint ? "True" : "False") << std::endl;
            }
#endif
        } else {
            // unsigned
            signedPrecision = false;
            if (outputLowValueIsNotZero) {
                hasZeroPoint = outputLowValueIsNotZero;
            }

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
            if (hasZeroPoint) {
                const float actual = quantizationDetails.outputLowValues[i] > 0.f ?
                    quantizationDetails.outputLowValues[i] :
                    quantizationDetails.outputHighValues[i];
                std::cout << "   actual: " << actual << ", threshold: 0.0" << std::endl;
                std::cout << "   hasZeroPoint: " << (hasZeroPoint ? "True" : "False") << std::endl;
            }
#endif
        }
    }

    if (!thereIsAtLeastOneNormalValue) {
        // all values are small and didn't define 'signedPrecision'
        signedPrecision = std::any_of(outputLowValues.begin(), outputLowValues.end(), [](const float& value) { return value < 0.f; });
        unsignedPrecision = !signedPrecision;
    }

    element::Type resultPrecision = element::undefined;
    // if zero point exists then result precision has to be defined by client code
    if (!hasZeroPoint) {
        if (signedPrecision && (!unsignedPrecision)) {
            switch (quantizationLevels) {
                case levels::int4:
                case levels::int8:
                case levels::int8_narrow_range:
                    resultPrecision = element::i8;
                    break;
                case levels::int16:
                case levels::int16_narrow_range:
                    resultPrecision = element::i16;
                    break;
                case levels::int32:
                case levels::int32_narrow_range:
                    resultPrecision = element::i32;
            }
        }

        if ((!signedPrecision) && unsignedPrecision) {
            switch (quantizationLevels) {
                case levels::int4:
                case levels::int8:
                case levels::int8_narrow_range:
                    resultPrecision = element::u8;
                    break;
                case levels::int16:
                case levels::int16_narrow_range:
                    resultPrecision = element::u16;
                    break;
                case levels::int32:
                case levels::int32_narrow_range:
                    resultPrecision = element::u32;
            }
        }
    }

    return LayerTransformation::PrecisionDetails(resultPrecision, hasNegative, hasZeroPoint);
}

LayerTransformation::PrecisionDetails LayerTransformation::getPrecisionDetails(const QuantizationDetails& quantizationDetails) {
    return getPrecisionDetails(quantizationDetails.levels, quantizationDetails.outputLowValues, quantizationDetails.outputHighValues);
}

bool LayerTransformation::isAsymmetricQuantization(const std::shared_ptr<const Node>& layer,
    const std::vector<ov::element::Type>& defaultPrecisions) {
    const auto nonConstNode = const_cast<ov::Node*>(layer.get())->shared_from_this();
    const auto dequantization = NetworkHelper::getDequantization(nonConstNode, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }
    return dequantization.subtract != nullptr;
}

bool LayerTransformation::isQuantized(const std::shared_ptr<const Node>& layer, const std::vector<ov::element::Type>& defaultPrecisions) const {
    return true;
}

DataPrecision LayerTransformation::getDataPrecision(
        const std::shared_ptr<Node>& layer,
        const QuantizationDetails& quantizationDetails,
        const std::vector<element::Type>& requiredPrecisions) {
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationInfo(layer);
#endif
    PrecisionDetails precisionDetailsAtOutputIntervals = getPrecisionDetails(quantizationDetails);

    if (precisionDetailsAtOutputIntervals.precision != element::undefined) {
        // FakeQuantize optimal precision not deined
        if (!requiredPrecisions.empty()) {
            const auto foundIt = std::find(requiredPrecisions.begin(), requiredPrecisions.end(), precisionDetailsAtOutputIntervals.precision);
            const element::Type resultPrecision = foundIt != requiredPrecisions.end() ?
                precisionDetailsAtOutputIntervals.precision :
                *requiredPrecisions.begin();

            if (!DataPrecision::check(resultPrecision, quantizationDetails.levels)) {
                return DataPrecision();
            }

            return DataPrecision(
                resultPrecision,
                DataPrecision::getMinValue(resultPrecision, quantizationDetails.levels),
                DataPrecision::getMaxValue(resultPrecision, quantizationDetails.levels),
                foundIt != requiredPrecisions.end() ? precisionDetailsAtOutputIntervals.hasZeroPoint : true);
        }
    } else {
        // FakeQuantize optimal precision is not deined
        if (!requiredPrecisions.empty()) {
            const element::Type resultPrecision = *requiredPrecisions.begin();

            if (!DataPrecision::check(resultPrecision, quantizationDetails.levels)) {
                return DataPrecision();
            }

            return DataPrecision(
                resultPrecision,
                DataPrecision::getMinValue(resultPrecision, quantizationDetails.levels),
                DataPrecision::getMaxValue(resultPrecision, quantizationDetails.levels),
                true);
        } else {
            // required precisions are not defined, not possible to get precision from FakeQuantize: something wrong
            // return not valid value
            return DataPrecision();
        }
    }

    // if required precisions is empty then use FakeQuantize optimal precision
    return DataPrecision(
        precisionDetailsAtOutputIntervals.precision,
        DataPrecision::getMinValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
        DataPrecision::getMaxValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
        precisionDetailsAtOutputIntervals.hasZeroPoint);
}

std::shared_ptr<ov::Node> LayerTransformation::moveDequantizationAfter(
    TransformationContext &context,
    const std::shared_ptr<ov::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updateOutputPrecision,
    const bool moveSubtract) const {
    OPENVINO_ASSERT(!dequantization.empty());
    const auto result = ov::pass::low_precision::NetworkHelper::moveDequantizationAfter(operation,
        dequantization,
        updateOutputPrecision,
        moveSubtract,
        defaultPrecisions);
    updateOutput(context, result.lastDequantization, result.newOperation);
    return result.newOperation;
}

std::shared_ptr<ov::Node> LayerTransformation::moveDequantizationBefore(
    TransformationContext& context,
    const std::shared_ptr<ov::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool moveSubtract) const {
    const auto result = ov::pass::low_precision::NetworkHelper::moveDequantizationBefore(operation,
        dequantization,
        moveSubtract);
    updateOutput(context, result.newOperation, result.lastDequantization);
    return result.newOperation;
}

bool LayerTransformation::updateOutput(
    TransformationContext &context,
    std::shared_ptr<ov::Node> lastNode,
    std::shared_ptr<ov::Node> originalNode) const {
    bool was_updated = false;
    for (auto output : lastNode->outputs()) {
        for (auto input : output.get_target_inputs()) {
            if (ov::is_type<ov::opset1::Result>(input.get_node())) {
                const std::string originalName = originalNode->get_friendly_name();
                originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
                lastNode->set_friendly_name(originalName);
                was_updated = true;
                break;
            }
        }
    }
    return was_updated;
}

void LayerTransformation::updateOutput(
    TransformationContext& context,
    std::shared_ptr<ov::Node> lastNode,
    std::string originalName) const {
    const size_t outputSize = context.model->get_output_size();
    for (size_t i = 0; i < outputSize; ++i) {
        std::shared_ptr<ov::Node> result = context.model->get_output_op(i);
        std::shared_ptr<ov::Node> outputNode = result->get_input_node_shared_ptr(0);
        if (outputNode.get() == lastNode.get()) {
            lastNode->set_friendly_name(originalName);
            break;
        }
    }
}

void LayerTransformation::addPattern(ov::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot) {
    MATCHER_SCOPE(SingleNodeMatcher);
    ov::graph_rewrite_callback internal_callback = [this, &context](ov::pass::pattern::Matcher &m) {
        const bool result = transform(context, m);
        (void)result;
#ifdef LPT_DISPLAY_PRECISION
        if (result) {
            auto operationNode = m.get_match_root();
            std::cout << "Operation was transformed: " <<
                operationNode->get_type_name() << ", " <<
                operationNode->get_friendly_name() << ", output operation precision: " <<
                ((operationNode->get_output_size() == 1u) ? operationNode->get_output_element_type(0) : ov::element::Type()) <<
                std::endl;
        }
#endif
        return false;
    };
    // TODO: better name for matcher? required?
    auto m = std::make_shared<ov::pass::pattern::Matcher>(patternRoot, matcher_name);
    auto match_pass = std::make_shared<ov::pass::MatcherPass>(
            m->get_name(),
            m,
            [m, internal_callback](const std::shared_ptr<Node>& node) -> bool {
                OPENVINO_DEBUG("Running matcher ", m->get_name(), " on ", node);
                OV_PASS_CALLBACK(m);
                if (std::dynamic_pointer_cast<ov::pass::pattern::Matcher>(m)->match(node->output(0))) {
                    OPENVINO_DEBUG("Matcher ", m->get_name(), " matched ", node);
                    bool status = internal_callback(*m.get());
                    // explicitly clear Matcher state because it holds pointers to matched nodes
                    m->clear_state();
                    return status;
                }
            m->clear_state();
            return false;
            },
            ov::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    pass.add_matcher(match_pass);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
