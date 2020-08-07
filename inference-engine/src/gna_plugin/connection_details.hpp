// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_layers.h"

namespace GNAPluginNS {
struct ConnectionDetails {
    InferenceEngine::CNNLayerPtr  input;
    bool needTransposeWeights = false;
    InferenceEngine::CNNLayerPtr permute;
    ConnectionDetails(InferenceEngine::CNNLayerPtr input,
        bool bTranspose = false,
        InferenceEngine::CNNLayerPtr permute = nullptr)
        : input(input)
        , needTransposeWeights(bTranspose)
        , permute(permute) {
    }
};
}  // namespace GNAPluginNS
