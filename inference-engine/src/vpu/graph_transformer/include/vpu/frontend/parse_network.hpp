// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <set>

#include <cpp/ie_cnn_network.h>
#include <details/caseless.hpp>

#include <vpu/frontend/stage_builder.hpp>
#include <vpu/model/model.hpp>
#include <vpu/custom_layer.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(LayersOrder,
    DFS,
    BFS)

class IeNetworkParser final {
//
// Public API
//
public:
    void clear();
    void checkNetwork(const ie::CNNNetwork& network);

    void parseNetworkBFS(const ie::CNNNetwork& network);
    void parseNetworkDFS(const ie::CNNNetwork& network);

    ie::InputsDataMap networkInputs;
    ie::OutputsDataMap networkOutputs;
    std::unordered_map<ie::DataPtr, ie::Blob::Ptr> constDatas;
    std::vector<ie::CNNLayerPtr> orderedLayers;
};

}  // namespace vpu
