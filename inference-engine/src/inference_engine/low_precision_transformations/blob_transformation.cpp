// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/blob_transformation.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <vector>


using namespace InferenceEngine;
using namespace InferenceEngine::details;

void BlobTransformation::transform(ICNNNetwork& network, bool transformWithFakeQuantizeOnWeights) const {
    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);

    for (const CNNLayerPtr& layer : layers) {
        if (layer->insData.size() < 2) {
            continue;
        }
        if (this->layersForTransformations.find(layer->type) == this->layersForTransformations.end()) {
            continue;
        }

        const CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(*layer, 1);
        if ((!transformWithFakeQuantizeOnWeights) &&
            ((weightsLayer->type == "FakeQuantize") || (weightsLayer->type == "Quantize"))) {
            continue;
        }

        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
        if (weightableLayer == nullptr) {
            continue;
        }

        const Blob::Ptr weightsBlob = CNNNetworkHelper::getWeights(*layer, false);
        if (weightsBlob != nullptr) {
            weightableLayer->blobs["weights"] = weightsBlob;
            weightableLayer->_weights = weightsBlob;
        }

        if (layer->insData.size() >= 3) {
            const Blob::Ptr biasesBlob = CNNNetworkHelper::getBiases(*layer);
            if (biasesBlob != nullptr) {
                weightableLayer->blobs["biases"] = biasesBlob;
                weightableLayer->_biases = biasesBlob;
            }

            CNNLayerPtr biasesLayer = CNNNetworkHelper::getParent(*layer, 2);
            CNNNetworkHelper::removeLayer(network, biasesLayer);
        }

        CNNNetworkHelper::removeLayer(network, weightsLayer);
    }
}
