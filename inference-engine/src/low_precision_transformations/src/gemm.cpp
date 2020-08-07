// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/gemm.hpp"

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ie_common.h>
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool GemmTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& gemm) const {
    if (!LayerTransformation::canBeTransformed(context, gemm)) {
        return false;
    }

    if ((gemm.insData.size() != 2) || (gemm.outData.size() != 1)) {
        THROW_IE_EXCEPTION << "layer outputs '" << gemm.outData.size() << "' is not correct";
    }

    const DataPtr inputData = gemm.insData[0].lock();
    if (inputData == nullptr) {
        return false;
    }

    const size_t inputChannelsCount = CNNNetworkHelper::getInputChannelsCount(gemm);
    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(gemm);

    const auto checkDequantizationLayer = [&](const CNNLayer& gemm, const size_t index) -> bool {
        if (parents.size() <= index) {
            return false;
        }
        const CNNLayerPtr scaleShift = parents[index];
        if (scaleShift->type != "ScaleShift") {
            return false;
        }

        std::vector<float> scales;
        std::vector<float> shifts;
        fillFromDequantizationLayer(*scaleShift, scales, shifts);

        if (scales.size() != inputChannelsCount) {
            return false;
        }
        if (std::any_of(scales.begin(), scales.end(), [&](const float value) { return value != scales[0]; })) {
            return false;
        }

        if (shifts.size() != inputChannelsCount) {
            return false;
        }
        if (std::any_of(shifts.begin(), shifts.end(), [&](const float value) { return value != 0.f; })) {
            return false;
        }

        return true;
    };

    if ((CNNNetworkHelper::getParents(gemm).size() != 2ul) ||
        (!checkDequantizationLayer(gemm, 0ul))) {
        return false;
    }

    if (parents[1]->type == "FakeQuantize") {
        if (!QuantizationDetails::isSupportedLevel(parents[1]->GetParamAsUInt("levels"))) {
            return false;
        }

        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*parents[1]);
        const DataPrecision dataPrecision = getDataPrecision(*parents[1], quantizationDetails, false, false);
        if (dataPrecision.precision == Precision::UNSPECIFIED) {
            return false;
        }
    }

    if (((parents[1]->type != "FakeQuantize") && (!checkDequantizationLayer(gemm, 1ul))) ||
        ((parents[1]->type == "FakeQuantize") && (!CNNNetworkHelper::onConstWeightsPath(*parents[1]) || !CNNNetworkHelper::onWeights(*parents[1])))) {
        return false;
    }

    return true;
}

void GemmTransformation::transform(TransformationContext& context, CNNLayer& gemm) const {
    if (!canBeTransformed(context, gemm)) {
        return;
    }

    if (!CaselessEq<std::string>()(gemm.type, "Gemm")) {
        THROW_IE_EXCEPTION << "layer '" << gemm.name << "' is not correct";
    }

    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(gemm);
    if (parents[1]->type == "FakeQuantize") {
        FullyConnectedTransformation::transform(context, gemm);
        return;
    }

    std::vector<float> originalDataDequantizationScales1;
    std::vector<float> originalDataDequantizationShifts1;
    fillFromDequantizationLayer(*parents[0], originalDataDequantizationScales1, originalDataDequantizationShifts1);
    std::vector<float> originalDataDequantizationScales2;
    std::vector<float> originalDataDequantizationShifts2;
    fillFromDequantizationLayer(*parents[1], originalDataDequantizationScales2, originalDataDequantizationShifts2);

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(gemm);
    std::vector<float> dequantizationScales(outputChannelsCount, originalDataDequantizationScales1[0] * originalDataDequantizationScales2[0]);
    std::vector<float> dequantizationShifts(outputChannelsCount, 0.f);

    CNNNetworkHelper::removeLayer(context.network, parents[0]);
    context.removeLayer(*parents[0]);

    if (parents[1]->type != "FakeQuantize") {
        CNNNetworkHelper::removeLayer(context.network, parents[1]);
        context.removeLayer(*parents[1]);
    }

    addDequantizationLayer(context, gemm, dequantizationScales, dequantizationShifts);
}

bool GemmTransformation::isQuantized(const CNNLayer& layer) const noexcept {
    // weightable layer version overriding
    return true;
}
