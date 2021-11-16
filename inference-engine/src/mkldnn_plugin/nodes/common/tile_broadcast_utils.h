// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_node.h"

#include <memory>
#include <vector>

class TileBroadcastCommon {
protected:
    static MKLDNNPlugin::VectorDims calculateDenseStrides(const MKLDNNPlugin::VectorDims &dims);
    std::vector<MKLDNNPlugin::NodeDesc> getSupportedConfigs(const MKLDNNPlugin::MKLDNNNode *node);
    bool prepareOptimizedParams(const MKLDNNPlugin::MKLDNNNode *node, MKLDNNPlugin::VectorDims& srcBlockedDims, MKLDNNPlugin::VectorDims& dstBlockedDims);

    void optimizedExecute(MKLDNNPlugin::MKLDNNNode *node);

    mutable MKLDNNPlugin::VectorDims repeats;
    bool optimizedCase = false;
    bool constMap[3] = { false };
    mutable bool needPrepareParamsVar = false;

private:
    static void fillOptimizedDimsAndSrcStrides(const MKLDNNPlugin::VectorDims &srcBlockedDims, const MKLDNNPlugin::VectorDims &blockedRepeats,
            MKLDNNPlugin::VectorDims &optimizedDims, MKLDNNPlugin::VectorDims &optimizedSrcStrides);

    static bool canBeExecutedInBlockedLayout(MKLDNNPlugin::VectorDims srcDims, MKLDNNPlugin::VectorDims repeats, const size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(MKLDNNPlugin::VectorDims srcDims, MKLDNNPlugin::VectorDims repeats);

    struct {
        MKLDNNPlugin::VectorDims dims;
        MKLDNNPlugin::VectorDims srcStrides;
        MKLDNNPlugin::VectorDims dstStrides;
        size_t copySize;
    } optimizedParams;
};
