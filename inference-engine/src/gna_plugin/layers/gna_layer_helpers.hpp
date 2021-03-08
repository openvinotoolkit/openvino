// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include "gna_layer_info.hpp"
#include "gna_plugin_log.hpp"

namespace GNAPluginNS {
namespace LayerUtils {
/**
 * @brief retrievs blob from const layer connected to certain layer
 * @param input
 * @param idx
 */
inline InferenceEngine::Blob::Ptr getParamFromInputAsBlob(InferenceEngine::CNNLayerPtr input, size_t idx) {
    if (input->insData.size() <= idx) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx << "input";
    }
    auto iLayerData = input->insData[idx].lock();
    if (!iLayerData) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                         << ", input: cannot dereference data weak-pointer";
    }
    auto iLayer = getCreatorLayer(iLayerData).lock();
    if (!iLayer) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                         << ", input: cannot dereference creator layer weak-pointer";
    }
    if (!LayerInfo(iLayer).isConst()) {
        THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                         << ", input: expected to be of type const, but was: " << iLayer->type;
    }

    if (!iLayer->blobs.count("custom")) {
        THROW_GNA_LAYER_EXCEPTION(iLayer) << "cannot get custom blob";
    }

    return iLayer->blobs["custom"];
}
}  // namespace LayerUtils
}  // namespace GNAPluginNS
