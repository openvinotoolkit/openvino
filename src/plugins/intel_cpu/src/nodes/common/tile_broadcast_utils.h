// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {

class TileBroadcastCommon {
protected:
    static VectorDims calculateDenseStrides(const VectorDims& dims);
    std::vector<NodeDesc> getSupportedConfigs(const Node* node, size_t outSize);
    bool prepareOptimizedParams(const Node* node, VectorDims& srcBlockedDims, VectorDims& dstBlockedDims);

    void optimizedExecute(const MemoryPtr& srcMemory, const MemoryPtr& dstMemory);

    VectorDims repeats;
    bool optimizedCase = false;
    bool constMap[3] = {false};
    mutable bool needPrepareParamsVar = false;

private:
    static void fillOptimizedDimsAndSrcStrides(const VectorDims& srcBlockedDims,
                                               const VectorDims& blockedRepeats,
                                               VectorDims& optimizedDims,
                                               VectorDims& optimizedSrcStrides);
    static void broadcastScalar(const char* srcData, char* dstData, size_t elt_cnt, size_t data_size);

    static bool canBeExecutedInBlockedLayout(VectorDims srcDims, VectorDims repeats, const size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(VectorDims srcDims, VectorDims repeats);

    struct {
        VectorDims dims;
        VectorDims srcStrides;
        VectorDims dstStrides;
        size_t copySize;
    } optimizedParams;
};

}  // namespace intel_cpu
}  // namespace ov
