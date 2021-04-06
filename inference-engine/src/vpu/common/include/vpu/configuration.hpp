// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <map>

#include "caseless.hpp"

#include "vpu/utils/optional.hpp"

namespace vpu {

struct CompilationConfig {
    std::string customLayers;

    bool detectBatch = true;

    bool enableEarlyEltwiseReLUFusion = true;

    //
    // Debug options
    //

    InferenceEngine::details::caseless_set<std::string> noneLayers;

    bool skipAllLayers() const {
        if (noneLayers.size() == 1) {
            const auto& val = *noneLayers.begin();
            return val == "*";
        }
        return false;
    }

    bool skipLayerType(const std::string& layerType) const {
        return noneLayers.count(layerType) != 0;
    }

    bool dumpAllPasses;

    bool disableReorder = false;  // TODO: rename to enableReorder and switch logic.
    bool disableConvertStages = false;
    bool checkPreprocessingInsideModel = true;
    bool enableCustomReshapeParam = false;

    //
    // Deprecated options
    //

    float inputScale = 1.0f;
    float inputBias = 0.0f;
};

}  // namespace vpu
