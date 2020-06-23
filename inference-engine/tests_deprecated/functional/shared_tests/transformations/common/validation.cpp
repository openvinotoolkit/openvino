// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "validation.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <unordered_set>

#include <details/caseless.hpp>
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/fake_quantize.hpp"
#include "low_precision_transformations/transformer.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

void LowPrecisionTransformationValidation::validate(
        CNNNetwork& network,
        // TODO: not correct, quantization parameters are defined per transformation
        const LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers,
        const std::vector<std::pair<std::string, std::string>>& originalLayersInfo) {
    validateIntervalsAndLevel(network, params, notTransformedLayers);
    validateWeightsToConst(network, params, notTransformedLayers);
    validatePrecision(network, params, notTransformedLayers);
    validateActivations(network, params, notTransformedLayers);
    validateScaleShifts(network, params, notTransformedLayers);
    validateConvolutions(network, params, notTransformedLayers);
    validateWithReference(network, originalLayersInfo);

    validateAsymmetricPattern(network, params, notTransformedLayers);

    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (layer->type == "Eltwise") {
            validateEltwise(network, params, *layer);
        }
    }

    // TODO: not ready
    // validateCustomLayerHandling(network, notTransformedLayers);
}

std::vector<std::pair<std::string, std::string>> LowPrecisionTransformationValidation::getLayers(const CNNNetwork& network) {
    std::vector<std::pair<std::string, std::string>> layerNames;
    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        layerNames.push_back(std::pair<std::string, std::string>(layer->name, layer->type));
    }
    return layerNames;
}

void LowPrecisionTransformationValidation::validateIntervalsAndLevel(
        const CNNNetwork& network,
        const LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers) {
    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (notTransformedLayers.find(layer->name) != notTransformedLayers.end()) {
            continue;
        }

        if (layer->type == "FakeQuantize") {
            const size_t levelsAsParam = layer->GetParamAsUInt("levels");
            QuantizeLayer* quantizeLayer = dynamic_cast<QuantizeLayer*>(layer.get());
            if (quantizeLayer == nullptr) {
                THROW_IE_EXCEPTION << "unexpected type";
            }

            if (levelsAsParam != quantizeLayer->levels) {
                THROW_IE_EXCEPTION << "level as param " << levelsAsParam << " is not equal level as member " << quantizeLayer->levels;
            }

            //// TODO: debug only
            //QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*layer);
            //std::cout << layer->name << (CNNNetworkHelper::onWeights(*layer) ? " on weights" : " on activations") <<
            //    ": levels=" << quantizationDetails.levels <<
            //    ": input [" << quantizationDetails.inputLowValues[0] << " - " << quantizationDetails.inputHighValues[0]
            //    << "], output [" << quantizationDetails.outputLowValues[0] << " - " << quantizationDetails.outputHighValues[0] << "]" << std::endl;
            bool multiBranch = false;

            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer, "Pooling");
            for (const CNNLayerPtr& child : children) {
                if ((child->type == "Eltwise") || (child->type == "Concat")) {
                    multiBranch = true;
                    break;
                }
            }

            validateFakeQuantize(layer, params, multiBranch);
        } else if (layer->type == "Eltwise") {
            // TODO: FQ on Eltwise specific logic is under development
        } else if (layer->type == "Concat") {
            // TODO: FQ on Concat specific logic is under development
        }
    }
}

void LowPrecisionTransformationValidation::validateWeightsToConst(
        const CNNNetwork& network,
        const LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers) {
    if ((!params.weightsToConst) ||
        (!std::any_of(
            params.precisionsOnActivations.begin(),
            params.precisionsOnActivations.end(),
            [](const Precision precision) { return precision == Precision::U8; }))) {
        return;
    }

    if ((!params.supportAsymmetricQuantization) &&
        (!std::any_of(params.precisionsOnWeights.begin(), params.precisionsOnWeights.end(), [](const Precision precision) { return precision.isSigned(); }))) {
        // U8 on weights in symmetric mode is ignored, shifts on weights are not supported
        return;
    }

    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if ((layer->type == "FakeQuantize") && CNNNetworkHelper::onWeights(*layer) && (layer->outData.size() == 1) &&
            (layer->outData[0]->getInputTo().begin()->second->type == "Convolution")) {
            CNNLayerPtr childLayer = CNNNetworkHelper::getChildren(*layer)[0];
            if (params.quantizeOutputs || (childLayer->outData[0]->getInputTo().size() != 0)) {
                ASSERT_TRUE(notTransformedLayers.find(childLayer->name) != notTransformedLayers.end()) <<
                    "FakeQuantize on weights was found: " << layer->name <<
                    " for layer " << childLayer->name;
            }
        }
    }
}

Precision getInputPrecision(const CNNLayer& layer) {
    if (layer.insData.size() < 1ul) {
        THROW_IE_EXCEPTION << "unexpected inputs count";
    }

    DataPtr layerParentData = layer.insData[0].lock();
    if (layerParentData == nullptr) {
        THROW_IE_EXCEPTION << "input data is nullable";
    }

    CNNLayerPtr layerParent = layerParentData->getCreatorLayer().lock();
    if (layerParent == nullptr) {
        THROW_IE_EXCEPTION << "parent is nullable";
    }

    if ((layer.type == "Convolution") && (layerParent->type == "Eltwise")) {
        DataPtr eltwiseParentData = layerParent->insData[0].lock();
        if (eltwiseParentData == nullptr) {
            THROW_IE_EXCEPTION << "Eltwise parent data is nullable";
        }

        // TODO: workaround for the first Convolution:
        // Issue-26622: [IE COMMON][LPT] Check if ScaleShift is dequantization ScaleShift(dequantizationLayersNames) before to apply transformation
        CNNLayerPtr eltwiseParent = eltwiseParentData->getCreatorLayer().lock();
        if (eltwiseParent->type == "Input") {
            return Precision::U8;
        }

        return eltwiseParentData->getTensorDesc().getPrecision();;
    } else {
        return layerParentData->getTensorDesc().getPrecision();
    }
}

Precision getOutputPrecision(const CNNLayer& layer) {
    if (layer.outData.size() < 1ul) {
        THROW_IE_EXCEPTION << "unexpected outputs count";
    }

    return layer.outData[0]->getTensorDesc().getPrecision();
}

// TODO: refactor (I8/U8 is used)
void LowPrecisionTransformationValidation::validatePrecision(
        const CNNNetwork& network,
        const LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers) {
    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (notTransformedLayers.find(layer->name) != notTransformedLayers.end()) {
            continue;
        }

        if ((!params.quantizeOutputs) && (layer->outData[0]->getInputTo().size() == 0ul)) {
            continue;
        }

        if (CaselessEq<std::string>()(layer->type, "FakeQuantize") && !isFakeQuantizeBeforeEltwiseOnConvolutionBranch(*layer)) {
            // TODO: handle if FakeQuantize on weights -> Const on weights transformation is disabled
            //if (CNNNetworkHelper::onWeights(*layer)) {
            //    for (const DataPtr data : layer->outData) {
            //        ASSERT_EQ(Precision::I8, data->getPrecision()) << "FakeQuantize out data on weights has unexpected precision";
            //    }
            //}

            if (!params.quantizeOutputs) {
                const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*layer, { "ScaleShift" });
                if ((children.size() == 0ul) || (children[0]->outData.size() == 0ul) || (children[0]->outData[0]->getInputTo().size() == 0ul)) {
                    continue;
                }
            }

            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
            bool hasDequantizationSS = false;
            for (const auto& child : children) {
                if (CaselessEq<std::string>()(child->type, "ScaleShift")) {
                    hasDequantizationSS = true;
                    break;
                }
            }

            if (params.updatePrecisions && hasDequantizationSS) {
                // while S8 is not supported on activations
                for (const DataPtr data : layer->outData) {
                    ASSERT_TRUE((data->getPrecision() == Precision::U8) || (data->getPrecision() == Precision::I8)) << "'" <<
                        layer->type << "', name '" <<
                        layer->name << "' out data on activations has unexpected precision " << data->getPrecision();
                }
            }
        } else if (layer->type == "Const") {
            if (CNNNetworkHelper::onWeights(*layer)) {
                // Note: Const layer on weights can has any original precision - check original network Const layer precision

                const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*layer, { "Eltwise" });
                if (children[0]->type == "FakeQuantize") {
                    // FakeQuantize on weights is possible if weights graph is complex
                    continue;
                }

                ASSERT_EQ(1ul, children.size()) <<
                    "children count " << children.size() <<
                    " is unexpected for " << layer->type << " '" << layer->name << "' layer on weights";
                ASSERT_TRUE((children[0]->type == "Convolution") || (children[0]->type == "FullyConnected") || (children[0]->type == "GEMM")) <<
                    "unexpected child type " << children[0]->type << " '" << children[0]->name << "' for layer " << layer->type << " '" << layer->name << "' on weights";

                if (children[0]->outData[0]->getInputTo().size() == 0) {
                    // output data precision depends on device
                    continue;
                }

                const Precision originalPrecision = getOutputPrecision(*children[0]);
                const Precision inputPrecision = getInputPrecision(*children[0]);
                const Precision weightsPrecision = inputPrecision == originalPrecision ? originalPrecision : params.precisionsOnWeights[0];

                if (inputPrecision != originalPrecision) {
                    ASSERT_TRUE((weightsPrecision == Precision::I8) || (weightsPrecision == Precision::U8)) <<
                        "unexpected weights precision " << weightsPrecision <<
                        " for " << children[0]->type << " " << children[0]->name;
                }

                for (auto it = layer->blobs.begin(); it != layer->blobs.end(); ++it) {
                    ASSERT_EQ(params.updatePrecisions ? weightsPrecision : originalPrecision, it->second->getTensorDesc().getPrecision()) <<
                        " constant layer on weights blob precison is not correct" <<
                        " for " << layer->type << " " << layer->name;;
                }

                for (const DataPtr data : layer->outData) {
                    ASSERT_EQ(params.updatePrecisions ? weightsPrecision : originalPrecision, data->getPrecision()) <<
                        " constant layer " << layer->name << " on weights blob precison is not correct";
                }
            }
        } else if ((layer->type == "Concat") || (layer->type == "Pooling")) {
            for (const DataPtr data : layer->outData) {
                if (params.updatePrecisions && (!CNNNetworkHelper::onWeights(*layer))) {
                    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*layer, { "Pooling" });
                    if (std::all_of(
                        parents.begin(),
                        parents.end(),
                        [](const CNNLayerPtr parent) { return (parent->type != "FakeQuantize") || QuantizationDetails::outputLayoutIsSupported(*parent); })) {
                        ASSERT_TRUE((data->getPrecision() == Precision::U8) || (data->getPrecision() == Precision::I8)) <<
                            layer->type << " layer, name '" <<
                            layer->name << "' out data has unexpected precision " << data->getPrecision();
                    }
                }
                // ASSERT_EQ(params.updatePrecisions ? Precision::U8 : Precision::FP32, data->getPrecision()) << " " << layer->type << " out data has unexpected precision " << data->getPrecision();
            }
        } else if ((layer->type == "Eltwise") || (layer->type == "Convolution")) {
            for (const DataPtr data : layer->outData) {
                // TODO: refactor: get original layer output precision from original network
                ASSERT_TRUE((data->getPrecision() == Precision::FP16) || (data->getPrecision() == Precision::FP32)) << "'" <<
                    layer->type << "', name '" <<
                    layer->name << "' out data has unexpected precision " << data->getPrecision();
            }
        }
    }
}

void LowPrecisionTransformationValidation::validateActivations(
    const CNNNetwork& network,
    const LayerTransformation::Params& params,
    const std::unordered_set<std::string>& notTransformedLayers) {
    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if ((notTransformedLayers.find(layer->name) != notTransformedLayers.end()) || (layer->type != "ReLU")) {
            continue;
        }

        const std::vector<CNNLayerPtr> reluParents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*layer, { "Pooling" });
        if ((reluParents.size() != 1) || (reluParents[0]->type != "ScaleShift")) {
            continue;
        }

        const CNNLayerPtr scaleShift = reluParents[0];

        const std::vector<CNNLayerPtr> scaleShiftParents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*scaleShift, { "Pooling" });
        // if Convolution is parent then ScaleShift can be generated by clean up transformation
        if ((scaleShiftParents.size() != 1) || (scaleShiftParents[0]->type == "Convolution")) {
            continue;
        }

        const float negativeSlope = layer->GetParamAsFloat("negative_slope", 0.0);
        if (negativeSlope != 0.0) {
            continue;
        }

        const Blob::Ptr weightsBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
        auto weights = CNNNetworkHelper::getFloatData(weightsBlob);
        const std::vector<float> scales = std::vector<float>(weights.get(), weights.get() + weightsBlob->size());

        const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
        auto biases = CNNNetworkHelper::getFloatData(biasesBlob);
        const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biasesBlob->size());

        if (!(std::equal(shifts.begin() + 1, shifts.end(), shifts.begin())) ||
            !(std::equal(scales.begin() + 1, scales.end(), scales.begin()))) {
            continue;
        }

        ASSERT_TRUE(true) << scaleShift->type << " '" << scaleShift->name << "' before " << layer->type << " '" << layer->name << "' was found";
    }
}

void LowPrecisionTransformationValidation::validateScaleShifts(
    const CNNNetwork& network,
    const LayerTransformation::Params& params,
    const std::unordered_set<std::string>& notTransformedLayers) {
    if (!params.updateBiases) {
        return;
    }

    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if ((notTransformedLayers.find(layer->name) != notTransformedLayers.end()) || (layer->type != "ScaleShift")) {
            continue;
        }

        const std::vector<CNNLayerPtr> scaleShiftParents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*layer, { "Pooling" });
        if ((scaleShiftParents.size() != 1) || (scaleShiftParents[0]->type != "Convolution")) {
            continue;
        }

        const Blob::Ptr biasesBlob = CNNNetworkHelper::getBlob(layer, "biases");
        auto biases = CNNNetworkHelper::getFloatData(biasesBlob);
        const std::vector<float> shifts = std::vector<float>(biases.get(), biases.get() + biasesBlob->size());

        ASSERT_TRUE(std::all_of(shifts.begin(), shifts.end(), [](float value) { return value == 0.0; })) <<
            layer->type << " '" << layer->name << "' after " <<
            scaleShiftParents[0]->type << " '" << scaleShiftParents[0]->name << "' has not zero shift values";
    }
}

void LowPrecisionTransformationValidation::validateConvolutions(
    const CNNNetwork& network,
    const LayerTransformation::Params& params,
    const std::unordered_set<std::string>& notTransformedLayers) {
    if (!params.updatePrecisions) {
        return;
    }

    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (layer->type != "Convolution") {
            continue;
        }

        CNNLayerPtr parent = CNNNetworkHelper::getParent(*layer, 0ul);
        const CNNLayerPtr precisionLayer = (parent->type == "Eltwise") ? parent : layer;
        const Precision precision = precisionLayer->insData[0].lock()->getTensorDesc().getPrecision();
        ASSERT_NE(Precision::I8, precision) << "unexpected input precision " << precision << " for " << layer->type << " " << layer->name;

        //std::cout << "LowPrecisionTransformationValidation::validateConvolutions: " << layer->type << " " << layer->name << ": " << precision << std::endl;
    }
}

void LowPrecisionTransformationValidation::validateWithReference(
    CNNNetwork& network,
    const std::vector<std::pair<std::string, std::string>>& originalLayersInfo) {
    std::unordered_map<std::string, CNNLayerPtr> layersMap;
    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        layersMap.emplace(layer->name, layer);
    }

    for (const auto layerInfo : originalLayersInfo) {
        const auto it = layersMap.find(layerInfo.first);

        // TODO: refactor: transformations move all ScaleShifts
        if (layerInfo.second == "ScaleShift") {
            continue;
        }

        // TODO: refactor: transformations can remove FakeQuantize and Const layers on weights
        if ((layerInfo.second == "FakeQuantize") || (layerInfo.second == "Const")) {
            continue;
        }

        if (it == layersMap.end()) {
            THROW_IE_EXCEPTION << "Layer '" << layerInfo.first << "' (" << layerInfo.second << ") is absent in transformed network";
            // std::cout << "Layer '" << layerInfo.first << "' (" << layerInfo.second << ") is absent in transformed network" << std::endl;
            // continue;
        }

        // TODO: last layer is ignored
        if ((it->second->outData.size() != 0) && (it->second->outData[0]->getInputTo().size() == 0)) {
            continue;
        }

        if (it->second->type != layerInfo.second) {
            THROW_IE_EXCEPTION << "Layer '" << layerInfo.first << "' (" << layerInfo.second << ") has unexpected type. Expected value " << it->second->type;
            // std::cout << "Layer '" << layerInfo.first << "' (" << layerInfo.second << ") has unexpected type. Expected value " << it->second->type << std::endl;
        }
    }
}

void LowPrecisionTransformationValidation::validateCustomLayerHandling(
    const CNNNetwork& network,
    const std::unordered_set<std::string>& notTransformedLayers) {
    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (layer->type == "FullyConnected") {
            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
            if ((children.size() == 0) || (children[0]->type != "ScaleShift")) {
                THROW_IE_EXCEPTION << "Layer " << layer->name << " is not handled";
            }
        }
    }
}

DataPrecision LowPrecisionTransformationValidation::getDataPrecision(const CNNLayer& layer, const LayerTransformation::Params& params) {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const bool onWeights = CNNNetworkHelper::onWeights(layer);

    if ((onWeights && (params.precisionsOnWeights.size() > 1ul)) ||
        ((!onWeights) && (params.precisionsOnActivations.size() > 1ul))) {
        const LayerTransformation::PrecisionDetails precisionDetails = FakeQuantizeTransformation(params).getPrecisionDetails(quantizationDetails);
        if (precisionDetails.precision != Precision::UNSPECIFIED) {
            const std::vector<Precision>& supportedPrecisions = onWeights ? params.precisionsOnWeights : params.precisionsOnActivations;
            const auto foundIt = std::find(supportedPrecisions.begin(), supportedPrecisions.end(), precisionDetails.precision);
            if (foundIt != supportedPrecisions.end()) {
                return DataPrecision(
                    precisionDetails.precision,
                    DataPrecision::getMinValue(precisionDetails.precision, quantizationDetails.levels),
                    DataPrecision::getMaxValue(precisionDetails.precision),
                    false);
            }
        }
    }

    const Precision precision = onWeights ? *params.precisionsOnWeights.begin() : *params.precisionsOnActivations.begin();
    return DataPrecision(
        precision,
        DataPrecision::getMinValue(precision, quantizationDetails.levels),
        DataPrecision::getMaxValue(precision),
        false);
}

// TODO: quantizedTensorAlignmentOnActivations is used
void LowPrecisionTransformationValidation::validateFakeQuantize(
    const CNNLayerPtr& layer,
    const LayerTransformation::Params& params,
    const bool multiBranch) {

    if (isFakeQuantizeBeforeEltwiseOnConvolutionBranch(*layer) || isFakeQuantizeBeforeConcat(*layer)) {
        return;
    }

    if (!params.quantizeOutputs) {
        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
        for (const CNNLayerPtr& child : children) {
            for (const DataPtr data : child->outData) {
                if (data->getInputTo().size() == 0ul) {
                    return;
                }
            }
        }
    }

    // TODO: Eltwise doesn't support assymetric quantization
    // TODO: make params per transformation
    // TODO: uncomment
    //if (params.supportAsymmetricQuantization) {
    //    if (CNNNetworkHelper::onWeights(*layer) && (params.precisionsOnWeights.size() == 1)) {
    //        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*layer);
    //        if (params.precisionsOnWeights.begin()->isSigned()) {
    //            ASSERT_TRUE(quantizationDetails.hasNegativeOutput());
    //        } else {
    //            ASSERT_FALSE(quantizationDetails.hasNegativeOutput());
    //        }
    //    } else if ((!CNNNetworkHelper::onWeights(*layer)) && (params.precisionsOnActivations.size() == 1)) {
    //        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*layer);
    //        if (params.precisionsOnActivations.begin()->isSigned()) {
    //            ASSERT_TRUE(quantizationDetails.hasNegativeOutput());
    //        } else {
    //            ASSERT_FALSE(quantizationDetails.hasNegativeOutput());
    //        }
    //    }
    //}

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*layer);
    // TODO: temporary fix: not possible to get min/max value for I8 if level was changed
    if (((quantizationDetails.levels != 255) && (quantizationDetails.levels != 256)) ||
        (!layer->outData.empty() &&
        // not quantized
        ((layer->outData[0]->getTensorDesc().getPrecision() == Precision::FP16) ||
        (layer->outData[0]->getTensorDesc().getPrecision() == Precision::FP32)))) {
        return;
    }

    const DataPrecision dataPrecision = getDataPrecision(*layer, params);
    for (size_t i = 0; i < quantizationDetails.outputLowValues.size(); ++i) {
        const auto lowValue = quantizationDetails.outputLowValues[i];
        const auto highValue = quantizationDetails.outputHighValues[i];

        if (((
                (params.quantizedTensorAlignmentOnActivations == LayerTransformation::QuantizedTensorAlignment::None) ||
                (params.quantizedTensorAlignmentOnActivations == LayerTransformation::QuantizedTensorAlignment::UpdateLevel)) &&
            ((!equals(dataPrecision.min, lowValue)) && (!equals(dataPrecision.max, highValue)))
            ) ||
            ((params.quantizedTensorAlignmentOnActivations == LayerTransformation::QuantizedTensorAlignment::UpdateIntervals) &&
            ((!equals(dataPrecision.min, lowValue)) || (!equals(dataPrecision.max, highValue))))
            ) {
            ASSERT_TRUE(true) <<
                "Output interval [" << lowValue << " - " << highValue <<
                "] for layer " << layer->name << " is not correct, " <<
                "expected [" << dataPrecision.min << " - " << dataPrecision.max << "]";

            //// TODO: debug only
            //std::cout <<
            //    "Output interval [" << lowValue << " - " << highValue <<
            //    "] for layer " << layer->name << " is not correct, " <<
            //    "expected [" << dataPrecision.min << " - " << dataPrecision.max << "]" << std::endl;
        }


        switch (params.quantizedTensorAlignmentOnActivations) {
            case LayerTransformation::QuantizedTensorAlignment::None: {
                if ((dataPrecision.precision == Precision::U8) || (dataPrecision.precision == Precision::I8)) {
                    if ((quantizationDetails.levels != 255) && (quantizationDetails.levels != 256)) {
                        ASSERT_TRUE(false) << "unexpected quantization levels " << quantizationDetails.levels <<
                            " for layer " << layer->name;
                    }
                } else {
                    ASSERT_TRUE(false) << "layer '" << layer->type << "', name '" << layer->name << "' has unexpected precision" << dataPrecision.precision;
                }

                break;
            }
            case LayerTransformation::QuantizedTensorAlignment::UpdateIntervals: {
                if ((dataPrecision.precision == Precision::U8) || (dataPrecision.precision == Precision::I8)) {
                    if ((quantizationDetails.levels != 255) && (quantizationDetails.levels != 256)) {
                        ASSERT_TRUE(false) << "unexpected quantization levels " << quantizationDetails.levels <<
                            " for layer " << layer->name;
                    }
                } else {
                    ASSERT_TRUE(false) << "layer '" << layer->type << "', name '" << layer->name << "' has unexpected precision" << dataPrecision.precision;
                }

                break;
            }
            case LayerTransformation::QuantizedTensorAlignment::UpdateLevel: {
                if ((dataPrecision.precision == Precision::U8) || (dataPrecision.precision == Precision::I8)) {
                    if (quantizationDetails.levels > 256) {
                        ASSERT_TRUE(false) << "layer '" << layer->type << "', name '" << layer->name << "' has unexpected quantization levels " << quantizationDetails.levels;
                    }

                    if (dataPrecision.precision == Precision::U8) {
                        if (quantizationDetails.outputLowValues[0] != 0.0) {
                            ASSERT_TRUE(false) << "unexpected output interval low value: " << quantizationDetails << " for layer " << layer->name;
                        }
                        if (quantizationDetails.levels != (quantizationDetails.outputHighValues[0] + 1)) {
                            ASSERT_TRUE(false) << "unexpected quantization levels " << quantizationDetails.levels <<
                                " for layer " << layer->name;
                        }
                    } else if (dataPrecision.precision == Precision::I8) {
                        // FIXME: alignment on weights is temporary unsupported
                        if (CNNNetworkHelper::onWeights(*layer)) {
                            break;
                        }

                        if (quantizationDetails.levels != (fabs(quantizationDetails.outputLowValues[0]) + quantizationDetails.outputHighValues[0] + 1)) {
                            ASSERT_TRUE(false) << "unexpected quantization levels " << quantizationDetails.levels << " for layer " << layer->name;
                        }
                    }
                } else {
                    ASSERT_TRUE(false) << "layer '" << layer->type << "', name '" << layer->name << "' has unexpected precision" << dataPrecision.precision;
                }
                break;
            }
            default: {
                THROW_IE_EXCEPTION << "unsupported QuantizedTensorAlignment mode";
            }
        }


        if (multiBranch) {
            if (((dataPrecision.precision == Precision::I8) || (dataPrecision.precision == Precision::U8)) &&
                (quantizationDetails.levels > 256)) {
                ASSERT_TRUE(false) << "unexpected quantization levels " << quantizationDetails.levels;
            }

            // TODO: FQ before Eltwise uses another algorithm - fix it
            //if ((lowValue < (dataPrecision.min - 0.0001)) || (highValue > (dataPrecision.max + 0.0001))) {
            //    ASSERT_TRUE(false) <<
            //        "Output interval [" << lowValue << " - " << highValue << "] for layer " << layer->name <<
            //        " is not included in [" << dataPrecision.min << " - " << dataPrecision.max << "]";

            //    //// TODO: debug only
            //    //std::cout <<
            //    //    "Output interval [" << lowValue << " - " << highValue << "] for layer " << layer->name <<
            //    //    " is not included in [" << dataPrecision.min << " - " << dataPrecision.max << "]" << std::endl;
            //}
        } else {
            if ((dataPrecision.precision == Precision::I8) || (dataPrecision.precision == Precision::U8)) {
                // FIXME: alignment on weights is temporary unsupported
                if (!CNNNetworkHelper::onWeights(*layer)) {
                    if ((dataPrecision.precision == Precision::U8) &&
                        ((!equals(dataPrecision.min, lowValue)) || (!equals(dataPrecision.max, highValue)))) {
                        ASSERT_TRUE(false) <<
                            "Output interval [" << lowValue << " - " << highValue <<
                            "] for layer " << layer->name << " is not correct, " <<
                            "expected [" << dataPrecision.min << " - " << dataPrecision.max << "]";
                    }
                }
            } else {
                ASSERT_TRUE(false) << "layer '" << layer->type << "', name '" << layer->name << "' has unexpected precision" << dataPrecision.precision;
            }
        }
    }
}

bool LowPrecisionTransformationValidation::isFakeQuantizeBeforeEltwiseOnConvolutionBranch(const CNNLayer& fakeQuantize) {
    // TODO: were is check on Convolution branch?
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(fakeQuantize);
    if (children.size() == 1lu) {
        if (CaselessEq<std::string>()(children[0]->type, "Eltwise"))
            return true;
        if (CaselessEq<std::string>()(children[0]->type, "ScaleShift")) {
            const std::vector<CNNLayerPtr> children2 = CNNNetworkHelper::getChildren(*children[0]);
            return (children2.size() == 1lu) && (CaselessEq<std::string>()(children2[0]->type, "Eltwise"));
        }
    }
    return false;
}

bool LowPrecisionTransformationValidation::isFakeQuantizeBeforeConcat(const CNNLayer& fakeQuantize) {
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(fakeQuantize, { "Pooling" });
    for (const CNNLayerPtr& child : children) {
        if (child->type == "Concat") {
            return true;
        }
    }
    return false;
}

bool inline LowPrecisionTransformationValidation::equals(const float value1, const float value2, const float max_diff) {
    return (std::fabs(value1 - value2) < max_diff);
}

void LowPrecisionTransformationValidation::validateEltwise(CNNNetwork& network, const LayerTransformation::Params& params, const CNNLayer& eltwise) {
    if (params.updatePrecisions) {
        // TODO: refactor: use used transformations to identify is Eltwise transformation or Eltwise CPU transformation used
        //const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(eltwise, { "Pooling", "ScaleShift" });
        //if ((parents[0]->type == "FakeQuantize") && (parents[1]->type == "FakeQuantize")) {
        //    const Precision precision0 = parents[0]->outData[0]->getPrecision();
        //    const Precision precision1 = parents[1]->outData[0]->getPrecision();
        //    if (
        //        (((precision0 != Precision::I8) && (precision0 != Precision::U8)) ||
        //        ((precision1 != Precision::FP32) && (precision1 != Precision::FP16))) &&
        //        (((precision0 != Precision::FP32) && (precision0 != Precision::FP16)) ||
        //        ((precision1 != Precision::I8) && (precision1 != Precision::U8)))
        //        ) {
        //        ASSERT_TRUE(false) << "layer precisions are not correct: " <<
        //            parents[0]->name << ", " << parents[0]->precision << " and " <<
        //            parents[1]->name << ", " << parents[1]->precision;
        //    }
        //}
    }
}

void LowPrecisionTransformationValidation::validateAsymmetricPattern(
    const CNNNetwork& network,
    const LayerTransformation::Params& params,
    const std::unordered_set<std::string>& notTransformedLayers) {
    const std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (const CNNLayerPtr layer : layers) {
        if (notTransformedLayers.find(layer->name) != notTransformedLayers.end()) {
            continue;
        }
        validateAsymmetricPattern(*layer, params);
    }
}

void LowPrecisionTransformationValidation::validateAsymmetricPattern(const CNNLayer& layer, const LayerTransformation::Params& params) {
    if (layer.type != "Convolution") {
        return;
    }

    if (params.supportAsymmetricQuantization && params.updatePrecisions) {
        CNNLayerPtr parentOnData = CNNNetworkHelper::getParent(layer, 0ul);
        if (parentOnData->type == "Eltwise") {
            validateAsymmetricPatternEltwise(*parentOnData, params);
        }

        CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(layer, 1ul);
        if (parentOnWeights == nullptr) {
            THROW_IE_EXCEPTION << "weights layer is absent for " << layer.type << " " << layer.name;
            // std::cout << "weights layer is absent for " << layer.type << " " << layer.name << std::endl;
            // return;
        }
        if (parentOnWeights->type == "Eltwise") {
            validateAsymmetricPatternEltwise(*parentOnWeights, params);
        }
    }
}

void LowPrecisionTransformationValidation::validateAsymmetricPatternEltwise(const CNNLayer& eltwise, const LayerTransformation::Params& params) {
    if ((!eltwise.CheckParamPresence("operation")) || (eltwise.GetParamAsString("operation") != "sub")) {
        return;
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(eltwise);
    for (const CNNLayerPtr& parent : parents) {
        if (parent->type == "Input") {
            return;
        }
    }

    // TODO: hardcoded for CPU
    const Precision precision = CNNNetworkHelper::onWeights(eltwise) ? Precision::I8 : Precision::U8;
    for (const CNNLayerPtr& parent : parents) {
        if (parent->type == "Const") {
            validateEmptyConst(*parent, params);
        }

        ASSERT_EQ(1, parent->outData.size());
        ASSERT_EQ(precision, parent->outData[0]->getPrecision()) <<
            "layer " << parent->type << " '" << parent->name <<
            "' has unexpected precision " << parent->outData[0]->getPrecision() <<
            ", expected: " << precision;
    }
}

void LowPrecisionTransformationValidation::validateEmptyConst(const CNNLayer& layer, const LayerTransformation::Params& params) {
    if (layer.type == "Const") {
        const Precision precision = layer.outData[0]->getTensorDesc().getPrecision();
        if (params.updatePrecisions) {
            // TODO: get correct precision here
            ASSERT_TRUE((precision == Precision::U8) || (precision == Precision::I8));
        } else {
            ASSERT_TRUE((precision == Precision::FP32) || (precision == Precision::FP16));
        }

        const auto it = layer.blobs.find("custom");
        ASSERT_NE(layer.blobs.end(), it);
        const Blob::Ptr blob = it->second;
        std::shared_ptr<float> buffer = CNNNetworkHelper::getFloatData(blob);
        ASSERT_TRUE(std::any_of(buffer.get(), buffer.get() + blob->size(), [](const float value) { return value != 0.0; })) <<
            layer.type << " layer '" << layer.name << "' has " << blob->getTensorDesc().getPrecision() << " zero values blob";
    }
}
