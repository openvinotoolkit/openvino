// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/layer_transformation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

#include <transformations/low_precision/network_helper.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

const char LayerTransformation::lastLayerPrefix[] = "_original";

LayerTransformation::LayerTransformation(const Params& params) :
    updatePrecisions(params.updatePrecisions),
    quantizeOutputs(params.quantizeOutputs),
    weightsToConst(params.weightsToConst),
    quantizedTensorAlignmentOnActivations(params.quantizedTensorAlignmentOnActivations),
    quantizedTensorAlignmentOnWeights(params.quantizedTensorAlignmentOnWeights),
    roundQuantizedValues(params.roundQuantizedValues),
    updateBiases(params.updateBiases),
    supportAsymmetricQuantization(params.supportAsymmetricQuantization),
    precisionsOnActivations(params.precisionsOnActivations),
    precisionsOnWeights(params.precisionsOnWeights),
    layerTransformationsManager(nullptr),
    paramsManager(nullptr),
    quantizationIntervalAsymmetryThreshold(2.e-4),
    zeroThreshold(1.e-6f),
    dequantizationShiftToZeroRatioTreshold(4.e-4f),
    minQuantizationLevels(2ul) {}

void LayerTransformation::setParamsManager(IParamsManager* paramsManager) noexcept {
    this->paramsManager = paramsManager;
}

void LayerTransformation::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    this->layerTransformationsManager = layerTransformationsManager;
}

void LayerTransformation::setUpdatePrecisions(const bool updatePrecisions) {
    this->updatePrecisions = updatePrecisions;
}

void LayerTransformation::setQuantizeOutputs(const bool quantizeOutputs) {
    this->quantizeOutputs = quantizeOutputs;
}

void LayerTransformation::setWeightsToConst(const bool weightsToConst) {
    this->weightsToConst = weightsToConst;
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

    if (!quantizeOutputs) {
        for (auto consumer : consumers(layer)) {
            if (consumer->get_type_info().is_castable(opset1::Result::get_type_info_static())) {
                return false;
            }
        }
    }
    return true;
}

void LayerTransformation::setQuantizationIntervalAsymmetryThreshold(const float value) {
    this->quantizationIntervalAsymmetryThreshold = value;
}

void LayerTransformation::setZeroThreshold(const float value) {
    this->zeroThreshold = value;
}

void LayerTransformation::setDequantizationShiftToZeroRatioTreshold(const float value) {
    this->dequantizationShiftToZeroRatioTreshold = value;
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
        const bool boundaryValuesAreNotZero =
            (std::fabs(quantizationDetails.outputLowValues[i]) >= zeroThreshold) &&
            (std::fabs(quantizationDetails.outputHighValues[i]) >= zeroThreshold);
        if (signedInterval && boundaryValuesAreNotZero) {
            // signed
            unsignedPrecision = false;
            hasNegative = true;

            const float expectedRatio = quantizationDetails.levels == 256 ? asymmetricIntervalSideRatio256 : -1.f;
            const float actualRatio = quantizationDetails.outputLowValues[i] / quantizationDetails.outputHighValues[i];
            const float actual = std::fabs(
                (actualRatio - expectedRatio) /
                std::max(fabs(quantizationDetails.outputLowValues[i]), fabs(quantizationDetails.outputHighValues[i])));
            if (actual > quantizationIntervalAsymmetryThreshold) {
                hasZeroPoint = true;
            }
        } else {
            // unsigned
            signedPrecision = false;
            if (boundaryValuesAreNotZero) {
                hasZeroPoint = boundaryValuesAreNotZero;
            }
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

bool LayerTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

DataPrecision LayerTransformation::getDataPrecision(
        std::shared_ptr<Node> layer,
        const QuantizationDetails& quantizationDetails,
        const bool onWeights,
        const bool supportAsymmetricQuantization) const {
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
                        DataPrecision::getMaxValue(resultPrecision),
                        foundIt != precisions.end() ? precisionDetailsAtOutputIntervals.hasZeroPoint : true);
                return dataPrecision;
            }
        }
    }

    const DataPrecision dataPrecision = precisions.empty() ?
                                        DataPrecision(element::undefined, 0.f, 0.f, false) :
                                        DataPrecision(
                                                *precisions.begin(),
                                                DataPrecision::getMinValue(*precisions.begin(), quantizationDetails.levels),
                                                DataPrecision::getMaxValue(*precisions.begin()),
                                                true);
    return dataPrecision;
}

void LayerTransformation::fillAvailablePrecisions(std::shared_ptr<Node> layer, std::vector<element::Type>& availablePrecisions) const {
    if (availablePrecisions.empty()) {
        return;
    }

    const std::vector<std::shared_ptr<Node>> children = consumers(layer);
    for (auto child : children) {
        if (child->get_type_info().is_castable(opset1::FakeQuantize::get_type_info_static())) {
            // FakeQuantize layer updates precision
            continue;
        }

        if (!layerTransformationsManager->isQuantized(child)) {
            // low precision chain is interrupted here: next layer supported precisions are ignored
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

std::shared_ptr<ngraph::Node> LayerTransformation::separateInStandaloneBranch(std::shared_ptr<ngraph::Node> node) const {
    FakeQuantizeDequantization dequantization = getDequantization(node);
    if (dequantization.isShared()) {
        bool nodeInputIndexToChangeWasFound = false;
        size_t nodeInputIndexToChange;
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            if (dequantization.multiply.get() == node->get_input_node_ptr(i)) {
                nodeInputIndexToChange = i;
                nodeInputIndexToChangeWasFound = true;
                break;
            }
        }
        if (!nodeInputIndexToChangeWasFound) {
            THROW_IE_LPT_EXCEPTION(*node) << " input index for " << dequantization.multiply->get_friendly_name() << " was not found";
        }

        std::shared_ptr<Node> parent = dequantization.data;
        if (dequantization.convert != nullptr) {
            parent = dequantization.convert->clone_with_new_inputs({ parent });
            parent->set_friendly_name(parent->get_name() + "_new");
        }

        if (dequantization.subtract != nullptr) {
            parent = dequantization.subtract->clone_with_new_inputs({
                parent,
                dequantization.subtract->input_value(1) });
            parent->set_friendly_name(parent->get_name() + "_new");
        }

        if (dequantization.multiply != nullptr) {
            parent = dequantization.multiply->clone_with_new_inputs({
                parent,
                dequantization.multiply->input_value(1) });
            parent->set_friendly_name(parent->get_name() + "_new");
        }

        auto newNode = node->clone_with_new_inputs({
            parent,
            node->input_value(1) });

        replace_node(node, newNode);
        newNode->set_friendly_name(node->get_friendly_name());

        return newNode;
    }

    return node;
}

void LayerTransformation::updateOutput(
    TransformationContext &context,
    std::shared_ptr<ngraph::Node> lastNode,
    std::shared_ptr<ngraph::Node> originalNode) const {
    const size_t outputSize = context.network->get_output_size();
    for (size_t i = 0; i < outputSize; ++i) {
        std::shared_ptr<ngraph::Node> result = context.network->get_output_op(i);
        std::shared_ptr<ngraph::Node> outputNode = result->get_input_node_shared_ptr(0);
        if (outputNode.get() == lastNode.get()) {
            const std::string originalName = originalNode->get_friendly_name();
            originalNode->set_friendly_name(originalName + "_original");
            lastNode->set_friendly_name(originalName);
            break;
        }
    }
}

void LayerTransformation::addPattern(ngraph::pass::GraphRewrite& pass, TransformationContext& context, std::shared_ptr<Node> patternRoot) const {
    ngraph::graph_rewrite_callback internal_callback = [this, &context](ngraph::pattern::Matcher &m) {
        transform(context, m);
        return true;
    };
    // TODO: better name for matcher? required?
    auto m = std::make_shared<ngraph::pattern::Matcher>(patternRoot, "SingleNodeMatcher");
    pass.add_matcher(m, internal_callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}


}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
