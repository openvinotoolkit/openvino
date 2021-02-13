// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include "permute_utils.h"

namespace MKLDNNPlugin {

class DepthAndSpaceUtils : public PermuteUtils {
protected:
    void prepareParams(const InferenceEngine::SizeVector& srcDims, const InferenceEngine::SizeVector& dstDims);
    void prepareOptimizedParams(const size_t nDims, const InferenceEngine::Precision precision);

    enum Mode {
        BLOCKS_FIRST = 0,
        DEPTH_FIRST = 1
    };

    Mode mode;
    size_t blockSize;
    size_t blockStep;

    struct {
        InferenceEngine::SizeVector shape5D;
        InferenceEngine::SizeVector block3D;
        size_t spatialStep;
        size_t batchStep;
        size_t srcChannels;
        size_t dstChannels;
        size_t blockShift;
        size_t channelShift;
    } params;
};

}  // namespace MKLDNNPlugin
