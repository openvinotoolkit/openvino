// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/concat_multi_channels.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <caseless.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <legacy/cnn_network_impl.hpp>
#include <legacy/ie_util_internal.hpp>

#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/quantization_details.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool isMultiChannel(const std::vector<CNNLayerPtr>& concatLayers) {
    for (const CNNLayerPtr& concat : concatLayers) {
        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(*concat, {"Pooling", "Resample"});
        if (CNNNetworkHelper::IsChild(children, {"Convolution"})) {
            return false;
        }
    }
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

    Subgraph subgraph = CNNNetworkHelper::getSubgraph(concat);
    if (subgraph.empty()) {
        return;
    }

    for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
        if (context.quantizedFakeQuantizeNames.find(quantizationLayer->name) != context.quantizedFakeQuantizeNames.end()) {
            return;
        }
    }

    if (!isMultiChannel(subgraph.concatLayers)) {
        ConcatTransformation::transform(context, concat);
        return;
    }

    // TODO: update later
    // TODO: check if precisions are different and return
    const DataPrecision dataPrecision = getDataPrecision(
        *subgraph.quantizationLayers[0],
        QuantizationDetails::getDetails(*subgraph.quantizationLayers[0]),
        false,
        false);
    if (dataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    std::unordered_map<std::string, std::vector<float>> dequantizationScalesLayers;
    std::unordered_map<std::string, std::vector<float>> dequantizationShiftsLayers;

    for (const CNNLayerPtr& fakeQuantizeLayer : subgraph.quantizationLayers) {
        if (fakeQuantizeLayer->type != "FakeQuantize") {
            continue;
        }

        const QuantizationDetails& quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);
        const size_t channelsCount = CNNNetworkHelper::getOutputChannelsCount(*fakeQuantizeLayer);
        std::vector<float> dequantizationScales(channelsCount);
        std::vector<float> dequantizationShifts(channelsCount);
        for (size_t i = 0ul; i < channelsCount; ++i) {
            dequantizationScales[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) / (dataPrecision.max - dataPrecision.min) :
                1.0;

            dequantizationShifts[i] = QuantizationDetails::isSupportedLevel(quantizationDetails.levels) ?
                (quantizationDetails.getOutputHighValue(i) - (quantizationDetails.getOutputHighValue(i) - quantizationDetails.getOutputLowValue(i)) *
                (dataPrecision.max / (dataPrecision.max - dataPrecision.min))) :
                0.f;
        }
        checkAndUpdateDequantizationShiftWithZero(quantizationDetails, dequantizationShifts);

        dequantizationScalesLayers[fakeQuantizeLayer->name] = dequantizationScales;
        dequantizationShiftsLayers[fakeQuantizeLayer->name] = dequantizationShifts;

        CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 3, dataPrecision.min);
        CNNNetworkHelper::updateBlobs(*fakeQuantizeLayer, 4, dataPrecision.max);
    }

    if (updatePrecisions) {
        for (const auto it : subgraph.layers) {
            const CNNLayer* layer = it.second;
            CNNNetworkHelper::setOutDataPrecision(*layer, dataPrecision.precision);
        }
    }

    auto dequantizationValuesCallback = [&](
        const CNNLayer& layer,
        const std::string originalLayerName,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts
        ) {
        if (layer.name != originalLayerName) {
            const auto update = [](
                const std::string& originalLayerName,
                const std::string& newLayerName,
                std::unordered_map<std::string, std::vector<float>>& dequantizationLayers) {
                auto it = dequantizationLayers.find(originalLayerName);
                if (it != dequantizationLayers.end()) {
                    dequantizationLayers.emplace(newLayerName, it->second);
                    dequantizationLayers.erase(it);
                }
            };
            update(originalLayerName, layer.name, dequantizationScalesLayers);
            update(originalLayerName, layer.name, dequantizationShiftsLayers);
        }

        fillDequantization(
            layer,
            dequantizationScalesLayers, dequantizationShiftsLayers,
            dequantizationScales, dequantizationShifts);
    };

    addDequantizationLayers(context, subgraph, dequantizationValuesCallback);

    for (const CNNLayerPtr& quantizationLayer : subgraph.quantizationLayers) {
        context.quantizedFakeQuantizeNames.insert(quantizationLayer->name);
    }
}

void ConcatMultiChannelsTransformation::fillDequantization(
    const CNNLayer& layer,
    const std::unordered_map<std::string, std::vector<float>>& dequantizationScalesLayers,
    const std::unordered_map<std::string, std::vector<float>>& dequantizationShiftsLayers,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) {
    std::vector<CNNLayerPtr> fakeQuantizes;
    if (layer.type == "FakeQuantize") {
        fakeQuantizes.push_back(std::make_shared<CNNLayer>(layer));
    } else {
        fillQuantization(layer, fakeQuantizes);
    }

    for (const CNNLayerPtr fakeQuantize : fakeQuantizes) {
        {
            const auto scalesIt = dequantizationScalesLayers.find(fakeQuantize->name);
            if (scalesIt == dequantizationScalesLayers.end()) {
                THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization scale values are not found";
            }
            const std::vector<float>& fakeQuantizeDequantizationScales = scalesIt->second;
            dequantizationScales.insert(dequantizationScales.end(), fakeQuantizeDequantizationScales.begin(), fakeQuantizeDequantizationScales.end());
        }
        {
            const auto shiftsIt = dequantizationShiftsLayers.find(fakeQuantize->name);
            if (shiftsIt == dequantizationShiftsLayers.end()) {
                THROW_IE_LPT_EXCEPTION(*fakeQuantize) << "dequantization shift values are not found";
            }
            const std::vector<float>& fakeQuantizeDequantizationShifts = shiftsIt->second;
            dequantizationShifts.insert(dequantizationShifts.end(), fakeQuantizeDequantizationShifts.begin(), fakeQuantizeDequantizationShifts.end());
        }
    }
}

void ConcatMultiChannelsTransformation::fillQuantization(const CNNLayer& layer, std::vector<CNNLayerPtr>& fakeQuantizes) {
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    for (const CNNLayerPtr parent : parents) {
        if (parent->type == "FakeQuantize") {
            fakeQuantizes.push_back(parent);
        } else {
            fillQuantization(*parent, fakeQuantizes);
        }
    }
}
