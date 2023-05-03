// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "ref_space_to_depth.hpp"

namespace ov {
namespace intel_cpu {

CommonSpaceToDepthExecutor::CommonSpaceToDepthExecutor(const ExecutorContext::CPtr context) : SpaceToDepthExecutor(context) {}

bool CommonSpaceToDepthExecutor::init(const SpaceToDepthAttrs &spaceToDepthAttrs,
                                      const std::vector<MemoryDescPtr> &srcDescs,
                                      const std::vector<MemoryDescPtr> &dstDescs,
                                      const dnnl::primitive_attr &attr) {
    commonSpaceToDepthAttrs = spaceToDepthAttrs;
    if (!one_of(spaceToDepthAttrs.layoutType,
                LayoutType::nCsp16c,
                LayoutType::nCsp8c,
                LayoutType::nspc,
                LayoutType::ncsp))
        IE_THROW() << "SpaceToDepth executor supports only 'nCsp16c', 'nCsp8c', "
                      "'nspc' or 'ncsp' layouts.";

    const bool isBlocked = one_of(spaceToDepthAttrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsFirst = spaceToDepthAttrs.layoutType == LayoutType::nspc;
    const auto& srcBlockedDims = spaceToDepthAttrs.srcBlockedDims;
    const auto& dstBlockedDims = spaceToDepthAttrs.destBlockedDims;

    size_t nDims = srcBlockedDims.size();

    const size_t reshapedRank =
            nDims + spaceToDepthAttrs.nSpatialDims + static_cast<int>(isBlocked && spaceToDepthAttrs.mode == SpaceToDepthAttrs::Mode::DEPTH_FIRST);
    const size_t lastIdx = reshapedRank - 1;
    size_t firstSpatialOrder = 2;

    PermuteParams params;
    params.data_size = spaceToDepthAttrs.dataSize;
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);
    params.src_block_dims[0] = srcBlockedDims[0];

    // reshaping of src dimensions and creating the permutation order for each layout:
    // new shape: [N, C, D1 / block_size, block_size, D2 / block_size, block_size, ... , DK / block_size, block_size]
    // order    : mode = blocks_first : [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K]
    //            mode = depth_first  : [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K]
    // where `k` is number of spatial dimensions

    auto reshapeAndSetPermOrder =
            [&](const size_t idx1, const size_t idx2, const size_t shift, const InferenceEngine::SizeVector& dims) {
                for (size_t i = 0; i < spaceToDepthAttrs.nSpatialDims; i++) {
                    params.order[i + idx1] = i * 2 + shift;
                    params.order[i + idx2] = i * 2 + shift + 1;

                    params.src_block_dims[params.order[i + idx1]] = dims[i + shift];
                    params.src_block_dims[params.order[i + idx2]] = spaceToDepthAttrs.blockSize;
                }
            };

    if (isBlocked) {
        size_t orderShiftForBlocks, orderShiftForDims;
        if (spaceToDepthAttrs.mode == SpaceToDepthAttrs::Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = spaceToDepthAttrs.nSpatialDims + 2;
            orderShiftForDims = 1;

            params.order[spaceToDepthAttrs.nSpatialDims + 1] = 1;
            params.order[lastIdx] = lastIdx;

            params.src_block_dims[params.order[spaceToDepthAttrs.nSpatialDims + 1]] = srcBlockedDims[1];
            params.src_block_dims[params.order[lastIdx]] = srcBlockedDims.back();
        } else {
            orderShiftForBlocks = 3;
            orderShiftForDims = spaceToDepthAttrs.nSpatialDims + 4;

            size_t extraBlockSize = srcBlockedDims.back() / spaceToDepthAttrs.blockStep;
            params.src_block_dims[1] = srcBlockedDims[1];
            params.src_block_dims[lastIdx] = extraBlockSize;
            params.src_block_dims[lastIdx - 1] = spaceToDepthAttrs.blockStep;

            params.order[1] = 1;
            params.order[2] = lastIdx - 1;
            params.order[lastIdx - spaceToDepthAttrs.nSpatialDims] = lastIdx;
        }

        reshapeAndSetPermOrder(orderShiftForBlocks, orderShiftForDims, firstSpatialOrder, dstBlockedDims);
    } else if (isChannelsFirst) {
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(spaceToDepthAttrs.mode == SpaceToDepthAttrs::DEPTH_FIRST) + spaceToDepthAttrs.nSpatialDims + 1;
        params.order[spaceToDepthAttrs.mode == SpaceToDepthAttrs::Mode::DEPTH_FIRST ? spaceToDepthAttrs.nSpatialDims + 1 : lastIdx] = lastIdx;
        params.src_block_dims[lastIdx] = srcBlockedDims.back();

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, dstBlockedDims);
    } else {
        size_t shift = static_cast<size_t>(spaceToDepthAttrs.mode == SpaceToDepthAttrs::DEPTH_FIRST) + 1;
        params.order[spaceToDepthAttrs.mode == SpaceToDepthAttrs::Mode::DEPTH_FIRST ? 1 : spaceToDepthAttrs.nSpatialDims + 1] = 1;
        params.src_block_dims[1] = srcBlockedDims[1];

        reshapeAndSetPermOrder(spaceToDepthAttrs.nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, dstBlockedDims);
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++)
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
    return true;
}

void CommonSpaceToDepthExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                      const int MB) {
    if (!permuteKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    permuteKernel->execute(reinterpret_cast<const uint8_t *>(src[0]->GetPtr()),
                           reinterpret_cast<uint8_t *>(dst[0]->GetPtr()), MB);
}

}   // namespace intel_cpu
}   // namespace ov