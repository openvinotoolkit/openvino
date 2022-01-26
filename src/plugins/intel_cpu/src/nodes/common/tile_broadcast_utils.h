// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_node.h"

#include <memory>
#include <vector>


namespace MKLDNNPlugin {

class TileBroadcastCommon {
protected:
    static VectorDims calculateDenseStrides(const VectorDims &dims);
    std::vector<NodeDesc> getSupportedConfigs(const MKLDNNNode *node);
    bool prepareOptimizedParams(const MKLDNNNode *node, VectorDims& srcBlockedDims, VectorDims& dstBlockedDims);

    void optimizedExecute(const MKLDNNMemoryPtr& srcMemory, const MKLDNNMemoryPtr& dstMemory);

    VectorDims repeats;
    bool optimizedCase = false;
    bool constMap[3] = { false };
    mutable bool needPrepareParamsVar = false;

private:
    static void fillOptimizedDimsAndSrcStrides(const VectorDims &srcBlockedDims, const VectorDims &blockedRepeats,
            VectorDims &optimizedDims, VectorDims &optimizedSrcStrides);
    static void broadcastScalar(const char *srcData, char *dstData, size_t elt_cnt, size_t data_size);

    static bool canBeExecutedInBlockedLayout(VectorDims srcDims, VectorDims repeats, const size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(VectorDims srcDims, VectorDims repeats);

    struct {
        VectorDims dims;
        VectorDims srcStrides;
        VectorDims dstStrides;
        size_t copySize;
    } optimizedParams;
};

}  // namespace MKLDNNPlugin
