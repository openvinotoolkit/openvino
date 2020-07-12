// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/layer_transformation.hpp>
#include <transformations/low_precision/network_helper.hpp>


#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>


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
        for (auto consumer : NetworkHelper::consumers(layer)) {
            if (consumer->get_type_info().is_castable(opset1::Result::get_type_info_static())) {
                return false;
            }
        }
    }
    return true;
}

#if 0 // TODO LPT-TO-NGRAPH

Precision LayerTransformation::getPrecisionBeforeParentDequantizationScaleShift(const CNNLayer& layer) {
    const CNNLayerPtr scaleShift = CNNNetworkHelper::getParent(layer, 0);
    if (scaleShift == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "dequantization ScaleShift layer is absent";
    }

    if (scaleShift->type != "ScaleShift") {
        THROW_TRANSFORMATION_EXCEPTION << "not expected dequantization layer type " << scaleShift->type;
    }

    if (scaleShift->insData.size() < 1) {
        THROW_TRANSFORMATION_EXCEPTION << "is not expected ScaleShift '" << scaleShift->name << "' insert data size "
                           << scaleShift->insData.size();
    }

    const DataWeakPtr insDataWeak = scaleShift->insData[0];
    const DataPtr insData = insDataWeak.lock();
    if (insData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "input data is absent";
    }

    return insData->getPrecision();
}
#endif

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

void LayerTransformation::printDequantizationInfo(const CNNLayer& layer) {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    std::cout <<
        layer.type << (CNNNetworkHelper::onWeights(layer) ? " on weights " : " on activations ") <<
        layer.name << ":" << std::endl <<
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

void LayerTransformation::addDequantizationLayer(
    TransformationContext& context,
    const std::shared_ptr<Node> layer,
    const FakeQuantizeDequantization& dequantization) const {
    //
}

void LayerTransformation::fillFromQuantizationDetails(
    const QuantizationDetails& quantizationDetails,
    const DataPrecision& dataPrecision,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    // TODO: refactor: make optional
    const float minQuantizationScale = 1e-32f;
    const float maxQuantizationScale = 1e32f;

    bool denormalOutputValuesWasUpdated = false;
    dequantizationScales.resize(quantizationDetails.outputChannelsCount);
    dequantizationShifts.resize(quantizationDetails.outputChannelsCount);

    for (size_t channel = 0lu; channel < quantizationDetails.outputChannelsCount; ++channel) {
        float dequantizationScale = 0.f;
        float dequantizationShift = 0.f;
        if (dataPrecision.precision.is_signed()) {
            // I8
            dequantizationScale =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);
            const float quantValue =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);

            const float actualLowPartQuantValue = std::fabs(quantizationDetails.getOutputLowValue(channel) / dataPrecision.min);
            const float actualHighPartQuantValue = std::fabs(quantizationDetails.getOutputHighValue(channel) / dataPrecision.max);

            if (dataPrecision.hasZeroPoint) {
                if (actualLowPartQuantValue < actualHighPartQuantValue) {
                    dequantizationShift = quantizationDetails.getOutputLowValue(channel) - dataPrecision.min * quantValue;
                } else {
                    dequantizationShift = quantizationDetails.getOutputHighValue(channel) - dataPrecision.max * quantValue;
                }
            }
        } else {
            // U8
            dequantizationScale =
                (quantizationDetails.getOutputHighValue(channel) - quantizationDetails.getOutputLowValue(channel)) /
                (dataPrecision.max - dataPrecision.min);
            if (dataPrecision.hasZeroPoint) {
                dequantizationShift = quantizationDetails.getOutputLowValue(channel);
            }
        }

        if (fabs(dequantizationScale) < minQuantizationScale) {
            dequantizationScales[channel] = minQuantizationScale;
            denormalOutputValuesWasUpdated = true;
        } else if (fabs(dequantizationScale) > maxQuantizationScale) {
            dequantizationScales[channel] = dequantizationScale > 0.f ? maxQuantizationScale : -maxQuantizationScale;
            denormalOutputValuesWasUpdated = true;
        } else {
            dequantizationScales[channel] = dequantizationScale;
        }

        dequantizationShifts[channel] = dequantizationShift;
    }
}

void LayerTransformation::checkAndUpdateDequantizationShiftWithZero(
    const QuantizationDetails& quantizationDetails,
    std::vector<float>& dequantizationShifts) const {
    auto compare = [](float value1, float value2) { return (std::fabs(value1) < std::fabs(value2)); };

    const auto maxShiftIt = std::max_element(dequantizationShifts.begin(), dequantizationShifts.end(), compare);
    if (maxShiftIt == dequantizationShifts.end()) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected dequantization shifts max value";
    }

    const auto maxOutputLowIt = std::max_element(quantizationDetails.outputLowValues.begin(), quantizationDetails.outputLowValues.end(), compare);
    if (maxOutputLowIt == quantizationDetails.outputLowValues.end()) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected dequantization output low value";
    }

    const auto maxOutputHighIt = std::max_element(quantizationDetails.outputHighValues.begin(), quantizationDetails.outputHighValues.end(), compare);
    if (maxOutputHighIt == quantizationDetails.outputHighValues.end()) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected dequantization output high value";
    }

    const float maxOutputIt = std::max(std::fabs(*maxOutputLowIt), std::fabs(*maxOutputHighIt));
    const float relative = std::fabs(*maxShiftIt) / std::fabs(maxOutputIt);
    if (relative < dequantizationShiftToZeroRatioTreshold) {
        std::fill(dequantizationShifts.begin(), dequantizationShifts.end(), 0.f);
    }
}

void LayerTransformation::fillFromDequantizationLayer(
    std::shared_ptr<Node> dequantizationLayer,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    // std::cerr << "[ ERROR ] NOT IMPLEMENTED METHOD IS CALLED " << __FILE__ << ":" << __LINE__ << "\n";
#if 0 // TODO LPT-TO-NGRAPH
    if (dequantizationLayer.type != "ScaleShift") {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected dequantization layer type " << dequantizationLayer.type;
    }

    CNNLayerPtr dequantizationLayerPtr = std::make_shared<CNNLayer>(dequantizationLayer);
    Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(dequantizationLayerPtr, "weights");
    const auto weightsBuffer = CNNNetworkHelper::getFloatData(weightsBlob);

    Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(dequantizationLayerPtr, "biases");
    const auto shiftsBuffer = CNNNetworkHelper::getFloatData(shiftsBlob);

    const size_t inputCannelsCount = CNNNetworkHelper::getInputChannelsCount(dequantizationLayer);
    dequantizationScales.resize(inputCannelsCount);
    dequantizationShifts.resize(inputCannelsCount);
    for (size_t channel = 0; channel < inputCannelsCount; ++channel) {
        dequantizationScales[channel] = (weightsBlob->size() == 1ul) ? weightsBuffer.get()[0] : weightsBuffer.get()[channel];
        dequantizationShifts[channel] = (shiftsBlob->size() == 1ul) ? shiftsBuffer.get()[0] : shiftsBuffer.get()[channel];
    }
#endif
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
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
            if (hasZeroPoint) {
                std::cout << "   actual: " << actual << ", threshold: " << quantizationIntervalAsymmetryThreshold << std::endl;
                std::cout << "   hasZeroPoint: " << (hasZeroPoint ? "True" : "False") << std::endl;
            }
#endif
        } else {
            // unsigned
            signedPrecision = false;
            if (boundaryValuesAreNotZero) {
                hasZeroPoint = boundaryValuesAreNotZero;
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

bool LayerTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

DataPrecision LayerTransformation::getDataPrecision(
        std::shared_ptr<Node> layer,
        const QuantizationDetails& quantizationDetails,
        const bool onWeights,
        const bool supportAsymmetricQuantization) const {
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
                        DataPrecision::getMaxValue(resultPrecision),
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
                                                DataPrecision::getMaxValue(*precisions.begin()),
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
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(node);
    if (dequantization.isShared()) {
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

        std::vector<Output<Node>> inputs = NetworkHelper::getInputs(node);
        const size_t inputIndex = NetworkHelper::getInputIndex(dequantization.multiply, node);
        inputs[inputIndex] = parent;
        const std::shared_ptr<Node> newNode = node->clone_with_new_inputs(inputs);

        replace_node(node, newNode);
        newNode->set_friendly_name(node->get_friendly_name());

        return newNode;
    }

    return node;
}

void LayerTransformation::moveDequantizationAfter(
    TransformationContext &context,
    const std::shared_ptr<ngraph::Node>& operation,
    const FakeQuantizeDequantization& dequantization,
    const bool updatePrecision) const {
    const auto result = ngraph::pass::low_precision::NetworkHelper::moveDequantizationAfter(operation, dequantization, updatePrecision);
    updateOutput(context, result.lastDequantization, result.newOperation);
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
