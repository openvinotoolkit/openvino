// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <details/caseless.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class NetworkConfig final {
public:
    void parse(const CompilationConfig& config);

    bool skipAllLayers() const;
    bool skipLayerType(const std::string& layerType) const { return _noneLayers.count(layerType) != 0; }

    bool hasManualDataScale() const { return !_dataScale.empty(); }
    const std::unordered_map<std::string, float>& dataScale() const { return _dataScale; }

    bool hwDisabled(const std::string& layerName) const;

private:
    ie::details::caseless_set<std::string> _noneLayers;

    std::unordered_map<std::string, float> _dataScale;

    std::unordered_set<std::string> _hwWhiteList;
    std::unordered_set<std::string> _hwBlackList;
};

}  // namespace vpu
