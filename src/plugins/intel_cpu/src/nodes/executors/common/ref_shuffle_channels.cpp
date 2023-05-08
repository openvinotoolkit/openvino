// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_shuffle_channels.hpp"

namespace ov {
namespace intel_cpu {

CommonShuffleChannelsExecutor::CommonShuffleChannelsExecutor(const ExecutorContext::CPtr context) : ShuffleChannelsExecutor(context) {}

bool CommonShuffleChannelsExecutor::init(const ShuffleChannelsAttributes &shuffleChannelsAttributes,
                                         const std::vector<MemoryDescPtr> &srcDescs,
                                         const std::vector<MemoryDescPtr> &dstDescs,
                                         const dnnl::primitive_attr &attr) {
    commonShuffleChannelsAttributes = shuffleChannelsAttributes;
    if (!one_of(shuffleChannelsAttributes.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c, LayoutType::nspc, LayoutType::ncsp))
        IE_THROW() << "ShuffleChannels executor supports only 'nCsp16c', 'nCsp8c', 'nspc' or 'ncsp' layouts.";

    const bool isBlocked = one_of(shuffleChannelsAttributes.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsLast = shuffleChannelsAttributes.layoutType == LayoutType::nspc;
    const auto& srcDims = shuffleChannelsAttributes.srcDims;
    const auto& srcBlockedDims = shuffleChannelsAttributes.srcBlockedDims;

    // 2 for decomposed axis dim, 1 for composed spatial dim
    const int batchRank = shuffleChannelsAttributes.axis;
    const int reshapedRank = batchRank + 2 + static_cast<int>(shuffleChannelsAttributes.spatialRank != 0)
            + static_cast<int>(isBlocked && (shuffleChannelsAttributes.spatialRank == 0));
    PermuteParams params;
    params.data_size = shuffleChannelsAttributes.dataSize;
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);

    const size_t groupSize = srcDims[shuffleChannelsAttributes.axis] / shuffleChannelsAttributes.group;
    size_t spatialShapeSize = 1;
    if (shuffleChannelsAttributes.spatialRank != 0) {
        for (int i = batchRank + 1; i < shuffleChannelsAttributes.dataRank; i++) {
            spatialShapeSize *= srcDims[i];
        }
    }

    auto decomposeAndTranpose = [&](int axis) {
        params.src_block_dims[axis] = shuffleChannelsAttributes.group;
        params.src_block_dims[axis + 1] = groupSize;
        params.order[axis] = axis + 1;
        params.order[axis + 1] = axis;
    };

    const int channelDim = 1;
    if (isBlocked) {
        size_t blkSize = srcBlockedDims.back();
        size_t CB = srcBlockedDims[1];
        if (shuffleChannelsAttributes.axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                params.order[i] = i;
                params.src_block_dims[i] = srcBlockedDims[i];
            }
            decomposeAndTranpose(batchRank);

            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize * blkSize;
        } else { // axis on batch
            decomposeAndTranpose(0);
            spatialShapeSize = CB * blkSize;
            for (int i = 2; i < shuffleChannelsAttributes.dataRank; i++) {
                spatialShapeSize *= srcDims[i];
            }
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else if (isChannelsLast) {
        if (shuffleChannelsAttributes.axis == channelDim) {  // axis on channel
            params.order[0] = 0;
            params.src_block_dims[0] = srcDims[0];
            params.order[1] = 1;
            params.src_block_dims[1] = spatialShapeSize;
            decomposeAndTranpose(2);
        } else if (shuffleChannelsAttributes.axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                if (i == 0) {
                    params.order[i] = i;
                    params.src_block_dims[i] = srcDims[i];
                } else if (i == 1) {
                    params.order[reshapedRank - 1] = reshapedRank - 1;
                    params.src_block_dims[params.order[reshapedRank - 1]] = srcDims[i];
                } else if (i > 1) {
                    params.order[i - 1] = i - 1;
                    params.src_block_dims[i - 1] = srcDims[i];
                }
            }
            decomposeAndTranpose(batchRank - 1);

            if (shuffleChannelsAttributes.spatialRank != 0) {
                params.order[batchRank + 1] = batchRank + 1;
                params.src_block_dims[batchRank + 1] = spatialShapeSize;
            }
        } else { // axis on batch
            decomposeAndTranpose(0);
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else {
        for (int i = 0; i < batchRank; i++) {
            params.src_block_dims[i] = srcDims[i];
            params.order[i] = i;
        }

        decomposeAndTranpose(batchRank);
        if (shuffleChannelsAttributes.spatialRank != 0) {
            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize;
        }
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++)
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
    return true;
}

void CommonShuffleChannelsExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                         const int MB) {
    if (!permuteKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    if (MB > 0)
        permuteKernel->execute(reinterpret_cast<const uint8_t *>(src[0]->GetPtr()),
                               reinterpret_cast<uint8_t *>(dst[0]->GetPtr()), MB);
    else
        permuteKernel->execute(reinterpret_cast<const uint8_t *>(src[0]->GetPtr()),
                               reinterpret_cast<uint8_t *>(dst[0]->GetPtr()));
}


}   // namespace intel_cpu
}   // namespace ov