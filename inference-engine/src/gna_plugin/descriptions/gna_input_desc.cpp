// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <frontend/quantized_layer_params.hpp>
#include <legacy/layer_transform.hpp>
#include <layers/gna_layer_info.hpp>

#include "gna_input_desc.hpp"
#include "gna_plugin_log.hpp"

using namespace InferenceEngine;
using namespace GNAPluginNS;

size_t InputDesc::minBytesRequiredForStoreInput(CNNLayerPtr layer) {
    auto quantized = getInjectedData<QuantizedLayerParams>(layer);
    size_t precision_bytes;
    if (quantized) {
        precision_bytes = 2;
    } else {
        precision_bytes = 4;
    }
    if (!LayerInfo(layer).isInput()) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "minBytesRequiredForStoreInput expect to worn on \"Input\" layer";
    }
    if (layer->outData.size() != 1) {
        THROW_GNA_LAYER_EXCEPTION(layer) << "minBytesRequiredForStoreInput invalid outData for the layer";
    }
    auto dims = layer->outData.front()->getTensorDesc().getDims();
    return details::product(dims.begin(), dims.end()) * precision_bytes;
}

std::vector<void *>& InputDesc::getPtrInputsGlobal(const std::string& name) {
    if (ptr_inputs_global_id.find(name) == ptr_inputs_global_id.end()) {
        ptr_inputs_global_storage.push_front({});
        ptr_inputs_global_id[name] = ptr_inputs_global_storage.begin();
    }
    return *ptr_inputs_global_id[name];
}

intel_dnn_orientation_t InputDesc::getOrientation(const std::string& name) {
    if (orientation_in.find(name) == orientation_in.end()) {
        THROW_GNA_EXCEPTION << "Can't find orientation for input name '" << name << "'";
    }
    return orientation_in[name];
}

float InputDesc::getScaleFactor(const std::size_t index) {
    if (index >= inputScaleFactors.size()) {
        THROW_GNA_EXCEPTION << "Can't find scale factor for index = " << index;
    }
    return inputScaleFactors[index];
}
