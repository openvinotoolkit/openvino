// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_multi_channels.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "network_serializer.h"

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

size_t getQuantizationLevel(const std::vector<CNNLayerPtr>& fakeQuantizeLayers) {
    size_t quantizationLevels = 0lu;
    for (int i = 0; i < fakeQuantizeLayers.size(); i++) {
        const CNNLayerPtr fakeQuantizeLayer = fakeQuantizeLayers[i];
        if (fakeQuantizeLayer->type != "FakeQuantize") {
            THROW_IE_EXCEPTION << "not expected layer type " << fakeQuantizeLayer->type;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);
        if (!QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {
            continue;
        }
        if (quantizationLevels == 0lu) {
            quantizationLevels = quantizationDetails.levels;
        } else if (quantizationLevels != quantizationDetails.levels) {
            THROW_IE_EXCEPTION << "different quantization levels " << quantizationLevels << " are not supported";
        }
    }

    return quantizationLevels;
}

bool isCascade(const std::vector<CNNLayerPtr>& concatLayers) {
    for (size_t index = 0ul; index < (concatLayers.size() - 1); ++index) {
        const CNNLayerPtr childConcatLayer = concatLayers[index];
        const CNNLayerPtr parentConcatLayer = concatLayers[index + 1];
        std::vector<CNNLayerPtr> parents =
            CNNNetworkHelper::getParentsRecursivelyExceptTypes(*childConcatLayer, {"Pooling"});

        bool parentConcatLayerWasFound = false;
        for (const CNNLayerPtr& parent : parents) {
            if (parent->name == parentConcatLayer->name) {
                parentConcatLayerWasFound = true;
                break;
            }
        }

        if (!parentConcatLayerWasFound) {
            return false;
        }
    }
    return true;
}

bool isMultiChannel(const std::vector<CNNLayerPtr>& concatLayers) {
    for (const CNNLayerPtr& concat : concatLayers) {
        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*concat, {"Pooling"});
        if (CNNNetworkHelper::IsChild(children, {"Convolution"})) {
            return false;
        }
    }
    return true;
}

bool ConcatMultiChannelsTransformation::getQuantizeLayers(
    CNNLayerPtr layer,
    std::vector<std::string>& childNameOurAfterQuantizeLayers,
    std::vector<CNNLayerPtr>& quantizeLayers,
    std::vector<std::vector<CNNLayerPtr>>& intermediateLayers,
    std::vector<CNNLayerPtr>& concatLayers,
    std::string childName,
    std::vector<CNNLayerPtr>& sideOutputLayers,
    std::vector<std::string>& childrenNameSideOutputLayers) {
    if (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
        !CaselessEq<std::string>()(layer->type, "Quantize")) {
        do {
            if (CaselessEq<std::string>()(layer->type, "Pooling")) {
                intermediateLayers.back().push_back(layer);
                std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*layer, {"Pooling"});
                std::string concatName;
                for (const CNNLayerPtr& child : children) {
                    if (child->type == "Concat") {
                        if (!concatName.empty()) {
                            THROW_IE_EXCEPTION << "several concat children layers are not supported";
                        }
                        concatName = child->name;
                    }
                }

                childName = concatName;
                layer = CNNNetworkHelper::getParent(*layer, 0);
            } else if (CaselessEq<std::string>()(layer->type, "Concat")) {
                concatLayers.push_back(layer);

                if (layer->outData[0]->getInputTo().size() != 1) {
                    sideOutputLayers.push_back(layer);
                    childrenNameSideOutputLayers.push_back(childName);
                }
                int size = layer->insData.size();
                childName = layer->name;
                for (int i = 0; i < size; i++) {
                    CNNLayerPtr layer1 = CNNNetworkHelper::getParent(*layer, i);
                    intermediateLayers.push_back({});
                    if (!getQuantizeLayers(
                        layer1,
                        childNameOurAfterQuantizeLayers,
                        quantizeLayers,
                        intermediateLayers,
                        concatLayers,
                        childName,
                        sideOutputLayers,
                        childrenNameSideOutputLayers)) {
                        return false;
                    }
                }
                return true;
            } else {
                return false;
            }
        } while (!CaselessEq<std::string>()(layer->type, "FakeQuantize") &&
                 !CaselessEq<std::string>()(layer->type, "Quantize"));
    }

    childNameOurAfterQuantizeLayers.push_back(childName);
    quantizeLayers.push_back(layer);
    return true;
}

void ConcatMultiChannelsTransformation::transform(TransformationContext& context, CNNLayer& concat) const {
    if (!canBeTransformed(context, concat)) {
        return;
    }

    if (!CaselessEq<std::string>()(concat.type, "Concat")) {
        THROW_IE_EXCEPTION << "layer type '" << concat.name << "' is not correct";
    }

    if ((concat.insData.size() < 2)) {
        THROW_IE_EXCEPTION << "layer inputs '" << concat.insData.size() << "' is not correct";
    }

    if (concat.GetParamAsUInt("axis", 1) != 1) {
        return;
    }

    std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(concat);
    if (CNNNetworkHelper::IsChild(children, {"Concat"}, {"Pooling"})) {
        return;
    }

    std::vector<CNNLayerPtr> quantizeLayers;
    std::vector<std::vector<CNNLayerPtr>> intermediateLayers;
    std::vector<CNNLayerPtr> concatLayers;
    std::vector<std::string> childNameOurAfterQuantizeLayers;
    std::vector<CNNLayerPtr> sideOutputLayers;
    std::vector<std::string> childrenNameSideOutputLayers;
    const auto inputDataNumber = concat.insData.size();
    for (size_t index = 0lu; index < inputDataNumber; index++) {
        DataPtr quantizeOnData = concat.insData[index].lock();
        if (quantizeOnData == nullptr) {
            THROW_IE_EXCEPTION << "input is absent";
        }
        auto parentLayer = quantizeOnData->getCreatorLayer().lock();
        intermediateLayers.push_back({});
        if (!getQuantizeLayers(
            parentLayer,
            childNameOurAfterQuantizeLayers,
            quantizeLayers,
            intermediateLayers,
            concatLayers,
            concat.name,
            sideOutputLayers,
            childrenNameSideOutputLayers)) {
            return;
        }
    }
    concatLayers.insert(concatLayers.begin(), std::make_shared<CNNLayer>(concat));

    if (quantizeLayers.empty()) {
        return;
    }

    for (const CNNLayerPtr& quantizeLayer : quantizeLayers) {
        if (!QuantizationDetails::outputLayoutIsSupported(*quantizeLayer)) {
            return;
        }
    }

    if ((!isCascade(concatLayers)) || (!isMultiChannel(concatLayers))) {
        ConcatTransformation::transform(context, concat);
        return;
    }

    // TODO: check if precisions are different and return
    std::vector<std::pair<CNNLayerPtr, std::vector<CNNLayerPtr>>> fakeQuantizeForConcatLayers;
    const DataPrecision dataPrecision = getDataPrecision(*quantizeLayers[0], QuantizationDetails::getDetails(*quantizeLayers[0]), false, false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    std::vector<float> finalDequantizationScales;
    std::vector<float> finalDequantizationShifts;
    const auto parentsCount = quantizeLayers.size();

    std::unordered_map<std::string, std::vector<float>> dequantizationScalesLayers;
    std::unordered_map<std::string, std::vector<float>> dequantizationShiftsLayers;

    for (int index = (concatLayers.size() - 1); index >= 0; --index) {
        const CNNLayerPtr concatLayer = concatLayers[index];

        const std::vector<CNNLayerPtr> parents =
            CNNNetworkHelper::getParentsRecursivelyExceptTypes(*concatLayer, {"Pooling"});
        for (const CNNLayerPtr& parent : parents) {
            if ((parent->type != "FakeQuantize") && (parent->type != "Concat")) {
                // TODO: handle
                THROW_IE_EXCEPTION << "layer type '" << parent->type << "' not supported";
            }
        }

        for (const CNNLayerPtr fakeQuantizeLayer : parents) {
            if (fakeQuantizeLayer->type != "FakeQuantize") {
                continue;
            }

            const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);
            const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(*fakeQuantizeLayer);
            std::vector<float> dequantizationScales(channelsCount);
            std::vector<float> dequantizationShifts(channelsCount);
            for (size_t i = 0ul; i < channelsCount; ++i) {
                dequantizationScales[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                    (quantizationDetails.outputHighValues[0] - quantizationDetails.outputLowValues[0]) / (dataPrecision.max - dataPrecision.min) :
                    1.0;

                dequantizationShifts[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                    (quantizationDetails.outputHighValues[0] - (quantizationDetails.outputHighValues[0] - quantizationDetails.outputLowValues[0]) *
                    (dataPrecision.max / (dataPrecision.max - dataPrecision.min))) :
                    0.f;
            }
            checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);

            finalDequantizationScales.insert(finalDequantizationScales.end(), dequantizationScales.begin(), dequantizationScales.end());
            finalDequantizationShifts.insert(finalDequantizationShifts.end(), dequantizationShifts.begin(), dequantizationShifts.end());

            dequantizationScalesLayers[fakeQuantizeLayer->name] = dequantizationScales;
            dequantizationShiftsLayers[fakeQuantizeLayer->name] = dequantizationShifts;

            if (QuantizationDetails::isSupportedLevel(quantizationDetails.levels)) {
                CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 3, dataPrecision.min);
                CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 4, dataPrecision.max);

                if (updatePrecisions) {
                    CNNNetworkHelper::setOutDataPrecision(*fakeQuantizeLayer, dataPrecision.precision);

                    const std::vector<CNNLayerPtr>& intermediateLayersList = intermediateLayers[index];
                    for (const CNNLayerPtr intermediateLayer : intermediateLayersList) {
                        CNNNetworkHelper::setOutDataPrecision(*intermediateLayer, dataPrecision.precision);
                    }
                }
            }
        }
    }

    // Add scaleshift at other outputs of the Quantize layer
    for (int index = 0; index < parentsCount; index++) {
        const CNNLayer& fakeQuantize = *quantizeLayers[index];
        context.quantizedFakeQuantizeNames.insert(fakeQuantize.name);

        std::vector<CNNLayerPtr> children =
            CNNNetworkHelper::getChildrenRecursivelyExceptTypes(fakeQuantize, {"Pooling"});
        for (auto it = children.begin(); it != children.end(); ++it) {
            CNNLayerPtr child = *it;
            if (index < childNameOurAfterQuantizeLayers.size()
                    && child->name == childNameOurAfterQuantizeLayers[index]) {
                children.erase(it, it + 1);
                break;
            }
        }

        for (const CNNLayerPtr& child : children) {
            const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*child);

            auto dequantizationScalesIt = dequantizationScalesLayers.find(fakeQuantize.name);
            if (dequantizationScalesIt == dequantizationScalesLayers.end()) {
                THROW_IE_EXCEPTION << "dequantization scales not found for layer " << fakeQuantize.name;
            }

            auto dequantizationShiftIt = dequantizationShiftsLayers.find(fakeQuantize.name);
            if (dequantizationShiftIt == dequantizationShiftsLayers.end()) {
                THROW_IE_EXCEPTION << "dequantization shifts not found for layer " << fakeQuantize.name;
            }

            const size_t fakeQuantizeOutputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(fakeQuantize);
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                quantizeLayers[index],
                child,
                DequantizationDetails(dequantizationScalesIt->second, dequantizationShiftIt->second, fakeQuantizeOutputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);

            if (updatePrecisions &&
                QuantizationDetails::isSupportedLevel(QuantizationDetails::getDetails(fakeQuantize).levels)) {
                CNNNetworkHelper::setOutDataPrecision(
                    CNNNetworkHelper::getLayers(fakeQuantize, *dequantizationLayer),
                    dataPrecision.precision);
            }
        }
    }

    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision(concat, dataPrecision.precision);
        for (const CNNLayerPtr& concatLayer : concatLayers) {
            if (concatLayer->name == concat.name) {
                continue;
            }

            // TODO: check if the same precision is used: U8 or S8 for all concat layers
            // TODO: fix & remove
            const DataPtr insData = concatLayer->insData[0].lock();
            if (insData == nullptr) {
                THROW_IE_LPT_EXCEPTION(*concatLayer) << "insert data is absent";
            }
            insData->setPrecision(dataPrecision.precision);
            // TODO: workaround
            concatLayer->precision = dataPrecision.precision;
            CNNNetworkHelper::setOutDataPrecision(*concatLayer, dataPrecision.precision);
        }
    }

    // Add scaleshift at outputs of our layers
    children = CNNNetworkHelper::getChildren(concat);
    if (children.size() == 0) {
        const std::string originalName = concat.name;
        CNNNetworkHelper::renameLayer(context.network, concat.name, concat.name + LayerTransformation::lastLayerPrefix);

        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(concat);
        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(concat),
            nullptr,
            DequantizationDetails(finalDequantizationScales, finalDequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(concat);
        for (const CNNLayerPtr& child : children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(concat),
                child,
                DequantizationDetails(finalDequantizationScales, finalDequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }

    // Add scaleshift at outputs of side branches
    for (int index = 0; index < sideOutputLayers.size(); index++) {
        const CNNLayerPtr concatLayer = sideOutputLayers[index];

        const size_t concatOutputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(*concatLayer);
        std::vector<float> dequantizationScales1(concatOutputChannelsCount);
        std::vector<float> dequantizationShifts1(concatOutputChannelsCount);
        for (size_t index_ = 0; index_ < concatOutputChannelsCount; ++index_) {
            dequantizationScales1[index_] = finalDequantizationScales[index_];
            dequantizationShifts1[index_] = finalDequantizationShifts[index_];
        }

        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*concatLayer, childrenNameSideOutputLayers[index]);
        for (int i = 0; i < children.size(); i++) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(*sideOutputLayers[index]),
                children[i],
                DequantizationDetails(dequantizationScales1, dequantizationShifts1, concatOutputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}
