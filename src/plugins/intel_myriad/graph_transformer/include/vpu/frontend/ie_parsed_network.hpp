// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <unordered_map>

#include <legacy/ie_layers.h>
#include <cpp/ie_cnn_network.h>

namespace vpu {

namespace ie = InferenceEngine;

struct IeParsedNetwork final {
    ie::InputsDataMap networkInputs;
    ie::OutputsDataMap networkOutputs;
    std::vector<std::pair<ie::DataPtr, ie::Blob::Ptr>> constDatas;
    std::vector<ie::CNNLayerPtr> orderedLayers;
};

IeParsedNetwork parseNetwork(const ie::CNNNetwork& network);

}  // namespace vpu
