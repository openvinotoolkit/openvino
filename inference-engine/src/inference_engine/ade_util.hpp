// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layers.h>

namespace ade {
class Graph;
}  // namespace ade

namespace InferenceEngine {

struct CNNLayerMetadata {
    CNNLayerPtr layer;

    static const char* name();
};

class ICNNNetwork;
void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& network);
}  // namespace InferenceEngine

