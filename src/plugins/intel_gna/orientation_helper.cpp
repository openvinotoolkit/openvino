// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "orientation_helper.hpp"

#include <legacy/ie_layers.h>

#include <gna_graph_tools.hpp>

namespace GNAPluginNS {
namespace helpers {

void updateModelInputOrientationWithoutConvolution(const InferenceEngine::CNNLayer& inputLayer,
                                                   const GNAPluginNS::backend::DnnComponents& components,
                                                   GNAPluginNS::GnaInputs& inputs) {
    // does not make sense to go further is there is no input to set
    auto input = inputs.find(inputLayer.name);

    if (input == inputs.end()) {
        return;
    }

    auto doesntHaveGnaMapping = [=](InferenceEngine::CNNLayerPtr l) {
        auto dnnLayer = components.findComponent(l);
        return dnnLayer == nullptr;
    };

    auto nextLayers = InferenceEngine::CNNNetGetAllNextLayersSkipCertain(&inputLayer, -1, doesntHaveGnaMapping);
    if (nextLayers.empty()) {
        input->orientation = kDnnInterleavedOrientation;
        return;
    }

    std::vector<intel_dnn_orientation_t> suggestedOrientations;

    auto dims = input->dims;
    auto raws = dims[0];
    auto rest = InferenceEngine::details::product(std::next(std::begin(dims)), std::end(dims));
    if (rest == 0) {
        rest = 1;
    }

    for (auto& nextLayer : nextLayers) {
        auto dnnLayer = components.findComponent(nextLayer);

        if (!dnnLayer) {
            THROW_GNA_LAYER_EXCEPTION(nextLayer) << " gna mapped layer search connection failed";
        }

        if (dnnLayer->operation != kDnnInterleaveOp && dnnLayer->operation != kDnnDeinterleaveOp && raws > 1 &&
            rest > 1) {
            suggestedOrientations.push_back(dnnLayer->orientation_in);
        }
    }

    if (suggestedOrientations.empty()) {
        input->orientation = kDnnNonInterleavedOrientation;
        return;
    }

    if (std::adjacent_find(suggestedOrientations.begin(),
                           suggestedOrientations.end(),
                           std::not_equal_to<intel_dnn_orientation_t>()) != suggestedOrientations.end()) {
        // unsupported case: orientations are different and they are important for these components
        THROW_GNA_EXCEPTION << "Input layer[" << inputLayer.name
                            << "] is used as input by multiple layers with different orientation!";
    }

    input->orientation = suggestedOrientations.front();
    return;
}

void updateModelOutputOrientation(const std::string& outputName,
                                  const std::string& cnnlayerName,
                                  const GNAPluginNS::backend::DnnComponents& components,
                                  GNAPluginNS::GnaOutputs& outputs) {
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
}  // namespace GNAPluginNS