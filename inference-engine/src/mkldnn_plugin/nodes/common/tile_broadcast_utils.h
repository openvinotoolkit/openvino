// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_node.h"

#include <memory>
#include <vector>

class TileBroadcastCommon {
protected:
    static InferenceEngine::SizeVector calculateStridesForDims(const InferenceEngine::SizeVector &dims);
    std::vector<MKLDNNPlugin::NodeDesc> getSupportedConfigs(MKLDNNPlugin::MKLDNNNode *node);
    bool prepareOptimizedParams(MKLDNNPlugin::MKLDNNNode *node, InferenceEngine::SizeVector& srcBlockedDims, InferenceEngine::SizeVector& dstBlockedDims);

    void optimizedExecute(MKLDNNPlugin::MKLDNNNode *node);

    mutable InferenceEngine::SizeVector repeats;
    bool optimizedCase = false;
    bool constMap[3] = { false };

private:
    static void fillOptimizedDimsAndSrcStrides(const InferenceEngine::SizeVector &srcBlockedDims, const InferenceEngine::SizeVector &blockedRepeats,
            InferenceEngine::SizeVector &optimizedDims, InferenceEngine::SizeVector &optimizedSrcStrides);

    static bool canBeExecutedInBlockedLayout(const MKLDNNPlugin::VectorDims& srcDims, const InferenceEngine::SizeVector& repeats, const size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(const MKLDNNPlugin::VectorDims& srcDims, const InferenceEngine::SizeVector& repeats);

    struct {
        std::vector<size_t> dims;
        std::vector<size_t> srcStrides;
        std::vector<size_t> dstStrides;
        size_t copySize;
    } optimizedParams;
};
