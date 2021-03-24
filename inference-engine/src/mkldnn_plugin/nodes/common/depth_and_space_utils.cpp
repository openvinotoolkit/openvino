// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_and_space_utils.h"

#include <ie_common.h>
#include <mkldnn_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

void DepthAndSpaceUtils::prepareParams(const SizeVector& srcDims, const SizeVector& dstDims) {
    size_t nDims = dstDims.size();
    for (size_t i = 0; i < nDims; ++i)
        params.shape5D.push_back(dstDims[i]);
    for (size_t i = nDims; i < 5; ++i)
        params.shape5D.push_back(1);

    for (size_t i = 0; i < nDims - 2; ++i)
        params.block3D.push_back(blockSize);
    for (size_t i = nDims - 2; i < 3; ++i)
        params.block3D.push_back(1);

    params.spatialStep = params.shape5D[2] * params.shape5D[3] * params.shape5D[4];
    params.batchStep = params.shape5D[1] * params.spatialStep;

    params.dstChannels = params.shape5D[1];
    params.srcChannels = params.dstChannels / blockStep;

    params.blockShift = mode == Mode::BLOCKS_FIRST ? params.srcChannels : 1;
    params.channelShift = mode == Mode::BLOCKS_FIRST ? 1 : blockStep;
}
