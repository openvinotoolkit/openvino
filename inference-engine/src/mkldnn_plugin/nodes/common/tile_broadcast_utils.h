// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "ie_parallel.hpp"
#include "cpu_memcpy.h"
#include <mkldnn_extension_utils.h>
#include <ngraph/runtime/host_tensor.hpp>
#include "mkldnn_node.h"

class TileBroadcastCommon {
protected:
    static InferenceEngine::SizeVector calculateStridesForDims(const InferenceEngine::SizeVector &dims);
    std::vector<MKLDNNPlugin::PrimitiveDescInfo> getSupportedConfigs(MKLDNNPlugin::MKLDNNNode *node);
    bool prepareOptimizedParams(MKLDNNPlugin::MKLDNNNode *node, InferenceEngine::SizeVector& srcBlockedDims, InferenceEngine::SizeVector& dstBlockedDims);

    void optimizedExecute(MKLDNNPlugin::MKLDNNNode *node);
    void ngraphExecute(MKLDNNPlugin::MKLDNNNode *node, std::shared_ptr<ngraph::Node> ngraphNode);

    InferenceEngine::SizeVector repeats;
    bool optimizedCase = false;

private:
    static void fillOptimizedDimsAndSrcStrides(const InferenceEngine::SizeVector &srcBlockedDims, const InferenceEngine::SizeVector &blockedRepeats,
            InferenceEngine::SizeVector &optimizedDims, InferenceEngine::SizeVector &optimizedSrcStrides);

    static bool canBeExecutedInBlockedLayout(const MKLDNNPlugin::MKLDNNDims& srcDims, const InferenceEngine::SizeVector& repeats, size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(MKLDNNPlugin::MKLDNNDims& srcDims, InferenceEngine::SizeVector& repeats);

    struct {
        std::vector<size_t> dims;
        std::vector<size_t> srcStrides;
        std::vector<size_t> dstStrides;
        size_t copySize;
    } optimizedParams;
};
