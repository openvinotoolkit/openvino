// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <string>
#include <unordered_set>

#include <ie_icnn_network.hpp>

namespace InferenceEngine {
namespace details {

class INFERENCE_ENGINE_API_CLASS(BlobTransformation) {
public:
    BlobTransformation() = default;
    void transform(ICNNNetwork& network, bool transformWithFakeQuantizeOnWeights = false) const;

private:
    const std::unordered_set<std::string> layersForTransformations = {
        "Convolution",
        "Deconvolution",
        "FullyConnected"
    };
};

}  // namespace details
}  // namespace InferenceEngine
