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
    int numSHAVEs = -1;
    int numCMXSlices = -1;
    int numExecutors = -1;
    int tilingCMXLimitKB = -1;

    bool hwOptimization = true;
    bool hwExtraSplit = false;

    std::string irWithVpuScalesDir;

    std::string customLayers;

    bool detectBatch = true;

    Optional<bool> injectSwOps;
    Optional<bool> packDataInCmx;
    bool mergeHwPoolToConv = true;
    bool hwDilation = false;
    bool forceDeprecatedCnnConversion = false;
    bool enableEarlyEltwiseReLUFusion = true;

    std::map<std::string, std::vector<int>> ioStrides;

    //
    // Debug options
    //

    InferenceEngine::details::caseless_set<std::string> hwWhiteList;
    InferenceEngine::details::caseless_set<std::string> hwBlackList;

    bool hwDisabled(const std::string& layerName) const {
        if (!hwWhiteList.empty()) {
            return hwWhiteList.count(layerName) == 0;
        }

        if (!hwBlackList.empty()) {
            return hwBlackList.count(layerName) != 0;
        }

        return false;
    }

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
    bool ignoreUnknownLayers = false;

    std::string dumpInternalGraphFileName;
    std::string dumpInternalGraphDirectory;
    bool dumpAllPasses;

    bool disableReorder = false;  // TODO: rename to enableReorder and switch logic.
    bool disableConvertStages = false;
    bool enablePermuteMerging = true;
    bool enableReplWithSCRelu = false;
    bool enableReplaceWithReduceMean = true;
    bool enableTensorIteratorUnrolling = false;
    bool forcePureTensorIterator = false;
    bool enableMemoryTypesAnnotation = false;
    bool enableWeightsAnalysis = true;
    bool checkPreprocessingInsideModel = true;
    bool enableCustomReshapeParam = false;

    //
    // Deprecated options
    //

    float inputScale = 1.0f;
    float inputBias = 0.0f;
};

}  // namespace vpu
