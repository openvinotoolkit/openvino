// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>

#include <legacy/ie_layers.h>
#include <ie_icnn_network.hpp>

namespace vpu {

namespace ie = InferenceEngine;

struct IeParsedNetwork final {
    ie::InputsDataMap networkInputs;
    ie::OutputsDataMap networkOutputs;
    std::unordered_map<ie::DataPtr, ie::Blob::Ptr> constDatas;
    std::vector<ie::CNNLayerPtr> orderedLayers;
};

IeParsedNetwork parseNetwork(const ie::ICNNNetwork& network);

}  // namespace vpu
