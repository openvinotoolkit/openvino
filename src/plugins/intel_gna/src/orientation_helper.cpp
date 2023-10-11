// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "orientation_helper.hpp"

#include <legacy/ie_layers.h>

#include <gna_graph_tools.hpp>

namespace ov {
namespace intel_gna {
namespace helpers {

void updateModelInputOrientationWithoutConvolution(const InferenceEngine::CNNLayer& inputLayer,
                                                   const backend::DnnComponents& components,
                                                   GnaInputs& inputs) {
    // does not make sense to go further is there is no input to set
    auto input = inputs.find(inputLayer.name);

    if (input == inputs.end()) {
        return;
    }

    auto dims = input->dims;
    if (dims.empty()) {
        // If input is scalar there is no sense to update orientation.
        return;
    }
    auto rowsNum = dims[0];

    auto doesntHaveGnaMapping = [=](InferenceEngine::CNNLayerPtr l) {
        auto dnnLayer = components.findComponent(l);
        return dnnLayer == nullptr;
    };

    auto nextLayers = InferenceEngine::CNNNetGetAllNextLayersSkipCertain(&inputLayer, -1, doesntHaveGnaMapping);
    if (nextLayers.empty()) {
        input->orientation = kDnnInterleavedOrientation;
        return;
    }

    auto columnProduct =
        std::accumulate(std::next(std::begin(dims)), std::end(dims), size_t{1}, std::multiplies<size_t>());

    // does not make sense to check if further if any of sizes is equal to 1
    // intput will be set to kDnnNonInterleavedOrientation
    if (rowsNum == 1 || columnProduct == 1) {
        input->orientation = kDnnNonInterleavedOrientation;
        return;
    }

    intel_dnn_orientation_t suggestedOrientation = kDnnUnknownOrientation;
    bool orientationIsSet = false;

    for (auto& nextLayer : nextLayers) {
        auto dnnLayer = components.findComponent(nextLayer);

        if (!dnnLayer) {
            THROW_GNA_LAYER_EXCEPTION(nextLayer) << " gna mapped layer search connection failed";
        }
        // Do not take Transposition operation intu consideration.
        if (dnnLayer->operation == kDnnInterleaveOp || dnnLayer->operation == kDnnDeinterleaveOp) {
            continue;
        }

        if (!orientationIsSet) {
            suggestedOrientation = dnnLayer->orientation_in;
            orientationIsSet = true;
        } else if (suggestedOrientation != dnnLayer->orientation_in) {
            // unsupported case: orientations are different and they are important for these components
            THROW_GNA_EXCEPTION << "Input layer[" << inputLayer.name
                                << "] is used as input by multiple layers with different orientation!";
        }
    }

    if (!orientationIsSet) {
        input->orientation = kDnnNonInterleavedOrientation;
        return;
    }

    input->orientation = suggestedOrientation;
    return;
}

void updateModelOutputOrientation(const std::string& outputName,
                                  const std::string& cnnlayerName,
                                  const backend::DnnComponents& components,
                                  GnaOutputs& outputs) {
    // if there is no output to set does not make sense to go further
    auto output = outputs.find(outputName);
    if (output == outputs.end()) {
        return;
    }

    auto dnnLayer = components.findComponent(cnnlayerName);
    if (dnnLayer && (dnnLayer->operation == kDnnInterleaveOp || dnnLayer->operation == kDnnDeinterleaveOp ||
                     dnnLayer->num_rows_out == 1 || dnnLayer->num_columns_out == 1)) {
        output->orientation = kDnnNonInterleavedOrientation;
    }
}
}  // namespace helpers
}  // namespace intel_gna
}  // namespace ov
