// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depth_to_space_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <cmath>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"

#define THROW_ERROR THROW_IE_EXCEPTION << "DepthToSpace layer with name '" << getName() << "' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNDepthToSpaceNode::MKLDNNDepthToSpaceNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNDepthToSpaceNode::getSupportedDescriptors() {
    auto* depthToSpaceLayer = dynamic_cast<DepthToSpaceLayer*>(getCnnLayer().get());
    if (depthToSpaceLayer == nullptr)
        THROW_ERROR << "cannot convert from CNN layer";

    SizeVector srcDims = depthToSpaceLayer->insData[0].lock()->getTensorDesc().getDims();
    if (srcDims.size() < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcDims.size() > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";

    SizeVector dstDims = depthToSpaceLayer->outData[0]->getTensorDesc().getDims();
    if (srcDims.size() != dstDims.size())
        THROW_ERROR << "has incorrect number of input/output dimensions";


    std::string modeString = depthToSpaceLayer->GetParamAsString("mode");
    if (modeString == "blocks_first") {
        mode = DepthToSpaceMode::BLOCKS_FIRST;
    } else if (modeString == "depth_first") {
        mode = DepthToSpaceMode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << modeString;
    }

    blockSize = depthToSpaceLayer->GetParamAsUInt("block_size", 1);
    if (blockSize == 0)
        THROW_ERROR << "Incorrect blockSize parameter is zero!";

    size_t nSpatialDims = srcDims.size() - 2;
    blockStep = static_cast<size_t>(std::pow(blockSize, nSpatialDims));
    if (srcDims[1] % blockStep)
        THROW_ERROR << "has block_size parameter which is incompatible with input tensor channels dimension size";

    if (srcDims[1] / blockStep != dstDims[1])
        THROW_ERROR << "has incompatible input/output channels";

    for (size_t i = 0; i < nSpatialDims; ++i) {
        if (srcDims[i + 2] * blockSize != dstDims[i + 2])
            THROW_ERROR << "has incompatible spatial dims";
    }

    if (getParentEdges().size() != 1)
        THROW_ERROR << "has incorrect number of input edges";
    if (getChildEdges().empty())
        THROW_ERROR << "has incorrect number of output edges";
}

void MKLDNNDepthToSpaceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();
    int nDims = srcDims.ToSizeVector().size();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    auto pushSupportedPrimitiveDescriptor = [&](const memory::format_tag memoryFormat) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, memoryFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, memoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memoryFormat});
    };

    auto canUseBlocked = [=](const size_t block) {
        return srcDims[1] % block == 0 && (mode == DepthToSpaceMode::BLOCKS_FIRST ?
                                           (srcDims[1] / block) % blockStep == 0 : block % blockStep == 0);
    };

    if (nDims == 4) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc);
        if (canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw8c);
        if (canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw16c);
    } else if (nDims == 5) {
        pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc);
        if (canUseBlocked(8))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw8c);
        if (canUseBlocked(16))
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw16c);
    }
    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
}

void MKLDNNDepthToSpaceNode::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    Precision precision = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision();
    optimizedParams.data_size = precision.size();

    SizeVector srcDims = getParentEdgeAt(0)->getBlob()->getTensorDesc().getDims();
    SizeVector dstDims = getChildEdgeAt(0)->getBlob()->getTensorDesc().getDims();

    size_t nDims = srcDims.size();
    for (size_t i = 0; i < nDims; ++i)
        params.shape5D.push_back(srcDims[i]);
    for (size_t i = nDims; i < 5; ++i)
        params.shape5D.push_back(1);

    for (size_t i = 0; i < nDims - 2; ++i)
        params.block3D.push_back(blockSize);
    for (size_t i = nDims - 2; i < 3; ++i)
        params.block3D.push_back(1);

    params.spatialStep = params.shape5D[2] * params.shape5D[3] * params.shape5D[4];
    params.batchStep = params.shape5D[1] * params.spatialStep;

    params.srcChannels = params.shape5D[1];
    params.dstChannels = params.srcChannels / blockStep;

    params.blockShift = mode == DepthToSpaceMode::BLOCKS_FIRST ? params.dstChannels : 1;
    params.channelShift = mode == DepthToSpaceMode::BLOCKS_FIRST ? 1 : blockStep;

    const bool isBlocked = getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat();
    size_t nSpatialDims = nDims - 2;
    nDims += nSpatialDims + static_cast<int>(isBlocked) + static_cast<int>(isBlocked && mode == DepthToSpaceMode::DEPTH_FIRST);
    size_t lastIdx = nDims - 1;

    order.resize(nDims);
    optimizedParams.src_block_dims.resize(nDims);
    order[0] = 0;
    optimizedParams.src_block_dims[0] = srcDims[0];

    if (isBlocked) {
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        SizeVector dstBlockedDims = getChildEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();

        size_t orderShiftForBlocks, orderShiftForDims;
        if (mode == DepthToSpaceMode::BLOCKS_FIRST) {
            orderShiftForBlocks = 1;
            orderShiftForDims = nSpatialDims + 2;

            optimizedParams.src_block_dims[nSpatialDims + 1] = srcBlockedDims[1] / blockStep;
            optimizedParams.src_block_dims[lastIdx] = srcBlockedDims.back();

            order[1] = nSpatialDims + 1;
            order[lastIdx] = lastIdx;
        } else {
            orderShiftForBlocks = nSpatialDims + 4;
            orderShiftForDims = 3;

            size_t newBlockSize = srcBlockedDims.back() / blockStep;
            size_t newBlocksCount = srcBlockedDims[1] * newBlockSize / srcBlockedDims.back();
            optimizedParams.src_block_dims[1] = newBlocksCount;
            optimizedParams.src_block_dims[2] = srcBlockedDims[1] / newBlocksCount;
            optimizedParams.src_block_dims[lastIdx - nSpatialDims] = newBlockSize;

            order[1] = 1;
            order[2] = 3;
            order[lastIdx - 1] = 2;
            order[lastIdx] = lastIdx - nSpatialDims;
        }

        for (size_t i = 0; i < nSpatialDims; i++) {
            optimizedParams.src_block_dims[i + orderShiftForDims] = srcBlockedDims[i + 2];
            optimizedParams.src_block_dims[i + orderShiftForBlocks] = blockSize;

            order[i * 2 + 2] = i + orderShiftForDims;
            order[i * 2 + 3] = i + orderShiftForBlocks;
        }
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        srcDims.push_back(srcDims[1]);
        dstDims.push_back(dstDims[1]);
        srcDims.erase(srcDims.begin() + 1);
        dstDims.erase(dstDims.begin() + 1);

        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + nSpatialDims + 1;
        order[lastIdx] = mode == DepthToSpaceMode::DEPTH_FIRST ? nSpatialDims + 1 : lastIdx;
        optimizedParams.src_block_dims[order[lastIdx]] = srcDims.back() / blockStep;

        for (size_t i = 0; i < nSpatialDims; i++) {
            optimizedParams.src_block_dims[i + shift] = blockSize;
            optimizedParams.src_block_dims[i + 1] = srcDims[i + 1];

            order[i * 2 + 1] = i + 1;
            order[i * 2 + 2] = i + shift;
        }
    } else {
        size_t shift = static_cast<size_t>(mode == DEPTH_FIRST) + 1;
        order[1] = mode == DEPTH_FIRST ? 1 : nSpatialDims + 1;
        optimizedParams.src_block_dims[order[1]] = srcDims[1] / blockStep;

        for (size_t i = 0; i < nSpatialDims; i++) {
            optimizedParams.src_block_dims[i + shift] = blockSize;
            optimizedParams.src_block_dims[i + nSpatialDims + 2] = srcDims[i + 2];

            order[i * 2 + 2] = i + nSpatialDims + 2;
            order[i * 2 + 3] = i + shift;
        }
    }

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
    optimizedParams.src_block_strides[lastIdx] = 1;
    optimizedParams.dst_block_strides[lastIdx] = 1;
    for (int i = lastIdx - 1; i >= 0; i--) {
        optimizedParams.src_block_strides[i] =
                optimizedParams.src_block_strides[i + 1] * optimizedParams.src_block_dims[i + 1];
        optimizedParams.dst_block_strides[i] =
                optimizedParams.dst_block_strides[i + 1] * optimizedParams.dst_block_dims[i + 1];
    }

    prepareConfigParams();
}

void MKLDNNDepthToSpaceNode::execute(mkldnn::stream strm) {
    auto srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    int MB = batchToProcess();
    if (params.shape5D[0] != MB)
        params.shape5D[0] = MB;

    if (permute_kernel) {
        auto &jcp = (*permute_kernel).jcp;
        if (jcp.dst_block_dims[0] != MB)
            jcp.dst_block_dims[0] = MB;

        optimizedExecute(srcData, dstData);
        return;
    }

    if (getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat()) {
        size_t dstCountBlocks = this->getSelectedPrimitiveDescriptor()->getConfig().outConfs[0].desc.getBlockingDesc().getBlockDims()[1];
        size_t block = this->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getBlockingDesc().getBlockDims().back();
        size_t blockRemainder = params.dstChannels % block;
        size_t lastBlock = blockRemainder == 0 ? block : blockRemainder;

        size_t srcBlock = block * params.shape5D[2] * params.shape5D[3] * params.shape5D[4];
        size_t dstBlock = block * params.shape5D[2] * params.block3D[0] * params.shape5D[3] * params.block3D[1] *
                          params.shape5D[4] * params.block3D[2];

        parallel_for2d(params.shape5D[0], params.shape5D[2], [&](size_t i0, size_t i2) {
            size_t srcIdx1 = i0 * params.batchStep + i2 * params.shape5D[3] * params.shape5D[4] * block;
            size_t dstIdx1 = i0 * dstBlock * dstCountBlocks;
            for (size_t b2 = 0; b2 < params.block3D[0]; ++b2) {
                size_t blk2 = b2 * params.block3D[1] * params.block3D[2] * params.blockShift;
                size_t blockNum2 = blk2 / block;
                size_t blockRemainder2 = blk2 - blockNum2 * block;

                size_t srcIdx2 = srcIdx1 + blockNum2 * srcBlock;
                size_t dstIdx2 = dstIdx1 + (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] *
                                 params.shape5D[4] * params.block3D[2] * block;
                for (size_t b3 = 0; b3 < params.block3D[1]; ++b3) {
                    size_t blk3 = blockRemainder2 + b3 * params.blockShift * params.block3D[2];
                    size_t blockNum3 = blk3 / block;
                    size_t blockRemainder3 = blk3 - blockNum3 * block;

                    for (size_t i3 = 0; i3 < params.shape5D[3]; ++i3) {
                        size_t srcIdx3 = srcIdx2 + i3 * params.shape5D[4] * block + blockNum3 * srcBlock;
                        size_t dstIdx3 = dstIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2] * block;
                        for (size_t b4 = 0; b4 < params.block3D[2]; ++b4) {
                            size_t blk4 = blockRemainder3 + b4 * params.blockShift;
                            size_t blockNum4 = blk4 / block;
                            size_t blockRemainder4 = blk4 - blockNum4 * block;

                            for (size_t i4 = 0; i4 < params.shape5D[4]; ++i4) {
                                size_t srcIdx4 = srcIdx3 + i4 * block + blockNum4 * srcBlock;
                                size_t dstIdx4 = dstIdx3 + (i4 * params.block3D[2] + b4) * block;
                                for (size_t i5 = 0; i5 < dstCountBlocks; ++i5) {
                                    size_t size = (i5 == dstCountBlocks - 1) ? lastBlock : block;
                                    for (size_t i6 = 0; i6 < size; ++i6) {
                                        size_t blk5 = blockRemainder4 + (i6 + i5 * block) * params.channelShift;
                                        size_t blockNum5 = blk5 / block;
                                        size_t blockRemainder5 = blk5 - blockNum5 * block;

                                        size_t srcIdx5 = srcIdx4 + blockRemainder5 + blockNum5 * srcBlock;
                                        size_t dstIdx5 = dstIdx4 + i6 + i5 * dstBlock;
                                        cpu_memcpy(dstData + dstIdx5 * optimizedParams.data_size,
                                                   srcData + srcIdx5 * optimizedParams.data_size,
                                                   optimizedParams.data_size);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        parallel_for2d(params.shape5D[0], params.shape5D[2], [&](size_t i0, size_t i2) {
            size_t srcIdx1 = i0 * params.batchStep;
            size_t dstIdx1 = i0 * params.batchStep;
            for (size_t b2 = 0; b2 < params.block3D[0]; b2++) {
                size_t srcIdx2 = srcIdx1 + i2 * params.shape5D[3] * params.shape5D[4] * params.srcChannels +
                        b2 * params.block3D[1] * params.block3D[2] * params.blockShift;
                size_t dstIdx2 = dstIdx1 +
                        (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] * params.shape5D[4] * params.block3D[2] * params.dstChannels;
                for (size_t i3 = 0; i3 < params.shape5D[3]; i3++) {
                    for (size_t b3 = 0; b3 < params.block3D[1]; b3++) {
                        size_t srcIdx3 = srcIdx2 + i3 * params.shape5D[4] * params.srcChannels + b3 * params.block3D[2] * params.blockShift;
                        size_t dstIdx3 = dstIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2] * params.dstChannels;
                        for (size_t i4 = 0; i4 < params.shape5D[4]; i4++) {
                            for (size_t b4 = 0; b4 < params.block3D[2]; b4++) {
                                size_t srcIdx4 = srcIdx3 + i4 * params.srcChannels + b4 * params.blockShift;
                                size_t dstIdx4 = dstIdx3 + (i4 * params.block3D[2] + b4) * params.dstChannels;
                                for (size_t i1 = 0; i1 < params.dstChannels; i1++) {
                                    size_t srcIdx5 = srcIdx4 + i1 * params.channelShift;
                                    size_t dstIdx5 = dstIdx4 + i1;
                                    cpu_memcpy(dstData + dstIdx5 * optimizedParams.data_size,
                                               srcData + srcIdx5 * optimizedParams.data_size,
                                               optimizedParams.data_size);
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        parallel_for2d(params.shape5D[0], params.dstChannels, [&](size_t i0, size_t i1) {
            size_t srcIdx1 = i0 * params.batchStep + i1 * params.channelShift * params.spatialStep;
            size_t dstIdx1 = i0 * params.batchStep + i1 * blockStep * params.spatialStep;
            for (size_t i2 = 0; i2 < params.shape5D[2]; i2++) {
                for (size_t b2 = 0; b2 < params.block3D[0]; b2++) {
                    size_t srcIdx2 = srcIdx1 + i2 * params.shape5D[3] * params.shape5D[4] +
                            b2 * params.block3D[1] * params.block3D[2] * params.blockShift * params.spatialStep;
                    size_t dstIdx2 = dstIdx1 + (i2 * params.block3D[0] + b2) * params.shape5D[3] * params.block3D[1] * params.shape5D[4] * params.block3D[2];
                    for (size_t i3 = 0; i3 < params.shape5D[3]; i3++) {
                        for (size_t b3 = 0; b3 < params.block3D[1]; b3++) {
                            size_t srcIdx3 = srcIdx2 + i3 * params.shape5D[4] + b3 * params.block3D[2] * params.blockShift * params.spatialStep;
                            size_t dstIdx3 = dstIdx2 + (i3 * params.block3D[1] + b3) * params.shape5D[4] * params.block3D[2];
                            for (size_t i4 = 0; i4 < params.shape5D[4]; i4++) {
                                for (size_t b4 = 0; b4 < params.block3D[2]; b4++) {
                                    size_t srcIdx4 = srcIdx3 + i4 + b4 * params.blockShift * params.spatialStep;
                                    size_t dstIdx4 = dstIdx3 + i4 * params.block3D[2] + b4;
                                    cpu_memcpy(dstData + dstIdx4 * optimizedParams.data_size,
                                               srcData + srcIdx4 * optimizedParams.data_size,
                                               optimizedParams.data_size);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}

bool MKLDNNDepthToSpaceNode::created() const {
    return getType() == DepthToSpace;
}
REG_MKLDNN_PRIM_FOR(MKLDNNDepthToSpaceNode, DepthToSpace);
