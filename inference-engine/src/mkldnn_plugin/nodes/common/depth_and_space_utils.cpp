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

void DepthAndSpaceUtils::prepareOptimizedParams(const size_t nDims, const Precision precision) {
    optimizedParams.dst_block_dims.resize(nDims);
    for (size_t i = 0; i < nDims; i++)
        optimizedParams.dst_block_dims[i] = optimizedParams.src_block_dims[order[i]];

    optimizedParams.src_block_order.resize(nDims);
    optimizedParams.dst_block_order.resize(nDims);
    for (size_t i = 0; i < nDims; i++) {
        optimizedParams.src_block_order[i] = i;
        optimizedParams.dst_block_order[i] = i;
    }

    optimizedParams.src_block_strides.resize(nDims);
    optimizedParams.dst_block_strides.resize(nDims);
    optimizedParams.src_block_strides[nDims - 1] = 1;
    optimizedParams.dst_block_strides[nDims - 1] = 1;
    for (int i = nDims - 2; i >= 0; i--) {
        optimizedParams.src_block_strides[i] =
                optimizedParams.src_block_strides[i + 1] * optimizedParams.src_block_dims[i + 1];
        optimizedParams.dst_block_strides[i] =
                optimizedParams.dst_block_strides[i + 1] * optimizedParams.dst_block_dims[i + 1];
    }

    optimizedParams.data_size = precision.size();
}
