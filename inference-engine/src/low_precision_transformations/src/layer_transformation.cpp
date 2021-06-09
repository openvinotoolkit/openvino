// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <low_precision/layer_transformation.hpp>
#include <low_precision/network_helper.hpp>


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

namespace ngraph {
namespace pass {
namespace low_precision {

const char LayerTransformation::originalLayerPostfix[] = "_original";

LayerTransformation::LayerTransformation(const Params& params) :
    updatePrecisions(params.updatePrecisions),
    quantizedTensorAlignmentOnActivations(params.quantizedTensorAlignmentOnActivations),
    quantizedTensorAlignmentOnWeights(params.quantizedTensorAlignmentOnWeights),
    supportAsymmetricQuantization(params.supportAsymmetricQuantization),
    precisionsOnActivations(params.precisionsOnActivations),
    precisionsOnWeights(params.precisionsOnWeights),
    deqPrecision(params.deqPrecision),
    support3DTensorOnActivations(params.support3DTensorOnActivations),
    deconvolutionSpecificChannelsRatio(params.deconvolutionSpecificChannelsRatio),
    quantizationIntervalAsymmetryThreshold(0.002f),
    zeroThreshold(1.e-6f),
    minQuantizationLevels(2ul),
    paramsManager(nullptr),
    layerTransformationsManager(nullptr) {}

void LayerTransformation::setParamsManager(IParamsManager* paramsManager) noexcept {
    this->paramsManager = paramsManager;
}

void LayerTransformation::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    this->layerTransformationsManager = layerTransformationsManager;
}

void LayerTransformation::setUpdatePrecisions(const bool updatePrecisions) {
    this->updatePrecisions = updatePrecisions;
}

void LayerTransformation::setQuantizedTensorAlignmentOnActivations(
    const QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    this->quantizedTensorAlignmentOnActivations = quantizedTensorAlignmentOnActivations;
}

void LayerTransformation::setQuantizedTensorAlignmentOnWeights(
    const QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    this->quantizedTensorAlignmentOnWeights = quantizedTensorAlignmentOnWeights;
}

const std::vector<element::Type>& LayerTransformation::getPrecisionsOnActivations() const {
    return precisionsOnActivations;
}

const std::vector<element::Type>& LayerTransformation::getPrecisionsOnWeights() const {
    return precisionsOnWeights;
}

bool LayerTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!isQuantized(layer)) {
        return false;
    }

    for (const auto& output : layer->outputs()) {
        const size_t size = output.get_shape().size();
        if ((size < 2ul) || (size > 5ul)) {
            return false;
        }
    }

    const auto dequantization = NetworkHelper::getDequantization(layer);
    if (!dequantization.empty()) {
        auto perChannelQuantization = [](const Shape dataShape, Shape constShape) {
            if ((dataShape.size() - constShape.size()) == 1ul) {
                constShape.insert(constShape.begin(), 1ul);
            }

            if ((constShape.size() >= 2ul) && (constShape[0] != 1ul)) {
                return false;
            }

            for (size_t i = 2; i < constShape.size(); ++i) {
                if (constShape[i] != 1ul) {
                    return false;
                }
            }
            return true;
        };

        if ((dequantization.subtract != nullptr) && (!perChannelQuantization(
            dequantization.subtract->output(0).get_shape(),
            dequantization.subtract->input(1).get_shape()))) {
            return false;
        }

        if ((dequantization.multiply != nullptr) && (!perChannelQuantization(
            dequantization.multiply->output(0).get_shape(),
            dequantization.multiply->input(1).get_shape()))) {
            return false;
        }
    }

    return true;
}

bool LayerTransformation::canBeTransformedSpatialDimension(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!isQuantized(layer)) {
        return false;
    }

    for (const auto& output : layer->outputs()) {
        const size_t size = output.get_shape().size();
        if ((size < 2ul) || (size > 5ul)) {
            return false;
        }
    }
    return true;
}

bool LayerTransformation::canSubtractBeHandled(const std::shared_ptr<Node>& op, const size_t parentIndex) const {
    return canSubtractBeHandled(op, NetworkHelper::getDequantization(op, parentIndex));
}

bool LayerTransformation::canSubtractBeHandled(const std::shared_ptr<Node>& op, const FakeQuantizeDequantization& dequantization) const {
    if (dequantization.empty() || (dequantization.subtract == nullptr)) {
        return true;
    }

    if (!supportAsymmetricQuantization) {
        return false;
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

    if (is_type<opset1::Constant>(parent)) {
        return true;
    } else if (is_type<opset1::Convert>(parent) && is_type<opset1::Constant>(parent->get_input_node_shared_ptr(0))) {
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
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(as_type_ptr<opset1::FakeQuantize>(layer));
    std::cout <<
        layer->get_type_name() << (NetworkHelper::isConstantPath(layer) ? " on weights " : " on activations ") <<
        layer->get_friendly_name() << ":" << std::endl <<
        "   details  : " << quantizationDetails << std::endl;
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

void LayerTransformation::setQuantizationIntervalAsymmetryThreshold(const float value) {
    this->quantizationIntervalAsymmetryThreshold = value;
}

void LayerTransformation::setZeroThreshold(const float value) {
    this->zeroThreshold = value;
}

void LayerTransformation::setMinQuantizationLevels(const size_t levels) {
    this->minQuantizationLevels = levels;
}

LayerTransformation::PrecisionDetails LayerTransformation::getPrecisionDetails(const QuantizationDetails& quantizationDetails) const {
    const float asymmetricIntervalSideRatio256 = -128.f / 127.f;
    bool hasNegative = false;
    bool signedPrecision = true;
    bool unsignedPrecision = true;

    bool hasZeroPoint = false;
    for (size_t i = 0; i < quantizationDetails.outputLowValues.size(); ++i) {
        const bool signedInterval = std::signbit(quantizationDetails.outputLowValues[i]) != std::signbit(quantizationDetails.outputHighValues[i]);
        const bool outputLowValueIsNotZero = std::fabs(quantizationDetails.outputLowValues[i]) >= zeroThreshold;
        if (signedInterval && outputLowValueIsNotZero) {
            // signed
            unsignedPrecision = false;
            hasNegative = true;

            if (quantizationDetails.outputHighValues[i] != 0.f) {
                const float expectedRatio = quantizationDetails.levels == 256 ? asymmetricIntervalSideRatio256 : -1.f;
                const float actualRatio = quantizationDetails.outputLowValues[i] / quantizationDetails.outputHighValues[i];
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

    if (!hasZeroPoint) {
        if (signedPrecision && (!unsignedPrecision)) {
            return LayerTransformation::PrecisionDetails(element::i8, hasNegative, hasZeroPoint);
        }

        if ((!signedPrecision) && unsignedPrecision) {
            return LayerTransformation::PrecisionDetails(element::u8, hasNegative, hasZeroPoint);
        }
    }

    return LayerTransformation::PrecisionDetails(element::undefined, hasNegative, hasZeroPoint);
}

bool LayerTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

DataPrecision LayerTransformation::getDataPrecision(
        std::shared_ptr<Node> layer,
        const QuantizationDetails& quantizationDetails,
        const bool onWeights) const {
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationInfo(layer);
#endif
    std::vector<element::Type> precisions = onWeights ? precisionsOnWeights : precisionsOnActivations;
    PrecisionDetails precisionDetailsAtOutputIntervals = getPrecisionDetails(quantizationDetails);
    {
        if (precisionDetailsAtOutputIntervals.precision != element::undefined) {
            if (!onWeights) {
                fillAvailablePrecisions(layer, precisions);
            }

            // if supportedPrecisions is empty then use the first available, not supported layer will be in original precision
            if (!precisions.empty()) {
                const auto foundIt = std::find(precisions.begin(), precisions.end(), precisionDetailsAtOutputIntervals.precision);
                const element::Type resultPrecision = foundIt != precisions.end() ?
                                                  precisionDetailsAtOutputIntervals.precision :
                                                  *precisions.begin();

                const DataPrecision dataPrecision(
                        resultPrecision,
                        DataPrecision::getMinValue(resultPrecision, quantizationDetails.levels),
                        DataPrecision::getMaxValue(resultPrecision, quantizationDetails.levels),
                        foundIt != precisions.end() ? precisionDetailsAtOutputIntervals.hasZeroPoint : true);

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
                printDequantizationInfo(dataPrecision);
#endif
                return dataPrecision;
            }
        }
    }

    const DataPrecision dataPrecision = precisions.empty() ?
                                        DataPrecision(element::undefined, 0.f, 0.f, false) :
                                        DataPrecision(
                                                *precisions.begin(),
                                                DataPrecision::getMinValue(*precisions.begin(), quantizationDetails.levels),
                                                DataPrecision::getMaxValue(*precisions.begin(), quantizationDetails.levels),
                                                true);
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationInfo(dataPrecision);
#endif
    return dataPrecision;
}

void LayerTransformation::fillAvailablePrecisions(std::shared_ptr<Node> layer, std::vector<element::Type>& availablePrecisions) const {
    if (availablePrecisions.empty()) {
        return;
    }

    const std::vector<std::shared_ptr<Node>> children = NetworkHelper::consumers(layer);
    for (auto child : children) {
        if (child->get_type_info().is_castable(opset1::FakeQuantize::get_type_info_static())) {
            // FakeQuantize layer updates precision
            continue;
        }

        if (!layerTransformationsManager->isQuantized(child)) {
            // low precision chain is interrupted here: next operation supported precisions are ignored
            continue;
        }

        const std::vector<element::Type> childPrecisionsOnActivations = paramsManager->getPrecisionsOnActivations(*child);
        if (childPrecisionsOnActivations.size() == 0ul) {
            continue;
        }

        for (size_t index = 0ul; index < availablePrecisions.size();) {
            const element::Type availablePrecision = availablePrecisions[index];
            if (!std::any_of(
                    childPrecisionsOnActivations.begin(),
                    childPrecisionsOnActivations.end(),
                    [&](const element::Type precision) { return availablePrecision == precision; })) {
                availablePrecisions.erase(availablePrecisions.begin() + index);
            } else {
                ++index;
            }
        }

        if (!layerTransformationsManager->isPrecisionPreserved(child)) {
            continue;
        }

        fillAvailablePrecisions(child, availablePrecisions);
        if (availablePrecisions.empty()) {
            return;
        }
    }
}

std::vector<std::shared_ptr<Node>> LayerTransformation::getChildrenRecursivelyExceptPrecisionPreserved(
        const std::shared_ptr<Node>& op) const noexcept {
    std::queue<std::shared_ptr<Node>> notHandledChildren;

    for (const auto& output : op->outputs()) {
        for (const auto& input : output.get_target_inputs()) {
            std::shared_ptr<Node> child = input.get_node()->shared_from_this();
            notHandledChildren.emplace(child);
        }
    }

    std::vector<std::shared_ptr<Node>> resultChildren;

    while (!notHandledChildren.empty()) {
        const std::shared_ptr<ngraph::Node> operation = notHandledChildren.front();
        notHandledChildren.pop();

        if (!this->layerTransformationsManager->isPrecisionPreserved(operation)) {
            resultChildren.push_back(operation);
            continue;
        }

        for (const auto& output : operation->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                std::shared_ptr<Node> child = input.get_node()->shared_from_this();
                notHandledChildren.emplace(child);
            }
        }
    }

    return resultChildren;
}

std::shared_ptr<ngraph::Node> LayerTransformation::moveDequantizationAfter(
    TransformationContext &context,
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updatePrecision,
    const bool moveSubtract) const {
    const auto result = ngraph::pass::low_precision::NetworkHelper::moveDequantizationAfter(operation, dequantization, updatePrecision, moveSubtract);
    updateOutput(context, result.lastDequantization, result.newOperation);
    return result.newOperation;
}

void LayerTransformation::updateOutput(
    TransformationContext &context,
    std::shared_ptr<ngraph::Node> lastNode,
    std::shared_ptr<ngraph::Node> originalNode) const {
    const size_t outputSize = context.function->get_output_size();
    for (size_t i = 0; i < outputSize; ++i) {
        std::shared_ptr<ngraph::Node> result = context.function->get_output_op(i);
        std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
        if (outputNode.get() == lastNode.get()) {
            const std::string originalName = originalNode->get_friendly_name();
            originalNode->set_friendly_name(originalName + LayerTransformation::originalLayerPostfix);
            lastNode->set_friendly_name(originalName);
            break;
        }
    }
}

void LayerTransformation::updateOutput(
    TransformationContext& context,
    std::shared_ptr<ngraph::Node> lastNode,
    std::string originalName) const {
    const size_t outputSize = context.function->get_output_size();
    for (size_t i = 0; i < outputSize; ++i) {
        std::shared_ptr<ngraph::Node> result = context.function->get_output_op(i);
        std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
        if (outputNode.get() == lastNode.get()) {
            lastNode->set_friendly_name(originalName);
            break;
        }
    }
}

void LayerTransformation::addPattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot) const {
    ngraph::graph_rewrite_callback internal_callback = [this, &context](ngraph::pattern::Matcher &m) {
        const bool result = transform(context, m);
        (void)result;
#ifdef LPT_DISPLAY_PRECISION
        if (result) {
            auto operationNode = m.get_match_root();
            std::cout << "Operation was transformed: " <<
                operationNode->get_type_name() << ", " <<
                operationNode->get_friendly_name() << ", output operation precision: " <<
                ((operationNode->get_output_size() == 1u) ? operationNode->get_output_element_type(0) : ngraph::element::Type()) <<
                std::endl;
        }
#endif
        return false;
    };
    // TODO: better name for matcher? required?
    auto m = std::make_shared<ngraph::pattern::Matcher>(patternRoot, "SingleNodeMatcher");
    NGRAPH_SUPPRESS_DEPRECATED_START
    pass.add_matcher(m, internal_callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
