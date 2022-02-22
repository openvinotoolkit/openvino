// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>

#include <ie_common.h>
#include <caseless.hpp>
#include "gna_graph_compiler.hpp"

namespace GNAPluginNS {
class LayersBuilder {
    using CreatorFnc = std::function<void(GNAGraphCompiler*, InferenceEngine::CNNLayerPtr)>;

public:
    LayersBuilder(const std::vector<std::string> &types, CreatorFnc callback) {
        for (auto && str : types) {
            getStorage()[str] = callback;
        }
    }
    static InferenceEngine::details::caseless_unordered_map<std::string, CreatorFnc> &getStorage() {
        static InferenceEngine::details::caseless_unordered_map<std::string, CreatorFnc> LayerBuilder;
        return LayerBuilder;
    }
};
}  // namespace GNAPluginNS
