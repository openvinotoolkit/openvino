// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_shuffle_channels_node.h"

#include <ie_parallel.hpp>
#include <mkldnn_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>

#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"

#include <string>
#include <cmath>

#define THROW_SHCH_ERROR IE_THROW() << "ShuffleChannels layer with name '" << getName() << "' "
#define CNTR_SIZE 3

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;

MKLDNNShuffleChannelsNode::MKLDNNShuffleChannelsNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNShuffleChannelsNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().empty() || getChildEdges().empty())
        THROW_SHCH_ERROR << "gets incorrect number of input/output edges.";

    if (getParentEdgeAt(0)->getDims().ndims() != getChildEdgeAt(0)->getDims().ndims()) {
        THROW_SHCH_ERROR << "gets mismatched input/output dimensions.";
    }

    auto *layer = getCnnLayer().get();
    dataDims = layer->outData[0]->getTensorDesc().getDims();
    dataRank = dataDims.size();

    axis = layer->GetParamAsInt("axis", 1);
    if (axis < 0)
        axis += dataRank;

    if (axis < 0 || axis >= static_cast<int>(dataRank))
        THROW_SHCH_ERROR << "gets incorrect axis number.";

    group = layer->GetParamAsUInt("group", 1);
    if (group == 0 || dataDims[axis] % group)
        THROW_SHCH_ERROR << "gets incorrect group parameter('group' parameter must evenly divide the channel dimension).";

    groupSize = dataDims[axis] / group;
}

void MKLDNNShuffleChannelsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (!mayiuse(cpu::x64::sse41)) {
        precision = Precision::FP32;
    }
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto srcDims = getParentEdgeAt(0)->getDims();
    int nDims = srcDims.ndims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    auto pushSupportedPrimitiveDescriptor = [&](const memory::format_tag memoryFormat) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), dataType, memoryFormat);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), dataType, memoryFormat);
        supportedPrimitiveDescriptors.push_back({config, impl_type, memoryFormat});
    };

    if (mayiuse(cpu::x64::sse41)) {
        if (nDims == 4) {
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nhwc);
            if (axis != 1) {
                if (impl_desc_type::jit_avx512 == impl_type) {
                    pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw16c);
                } else {
                    pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nChw8c);
                }
            }
        } else if (nDims == 5) {
            pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::ndhwc);
            if (axis != 1) {
                if (impl_desc_type::jit_avx512 == impl_type) {
                    pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw16c);
                } else {
                    pushSupportedPrimitiveDescriptor(mkldnn::memory::format_tag::nCdhw8c);
                }
            }
        }
    }

    pushSupportedPrimitiveDescriptor(MKLDNNMemory::GetPlainFormat(getParentEdgeAt(0)->getDims()));
}

void MKLDNNShuffleChannelsNode::createPrimitive() {
    if (prim || !mayiuse(cpu::x64::sse41))
        return;
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_SHCH_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_SHCH_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_SHCH_ERROR << "has unidentified preferable primitive descriptor";

    const bool isBlocked = getParentEdgeAt(0)->getMemory().GetDesc().isBlockedCFormat();

    int batchRank = axis;
    int spatialRank = dataRank - axis - 1;

    // 2 for decompose axis dim, 1 for composed spatial dim
    int reshapedRank = batchRank + 2 + static_cast<int>(spatialRank != 0) + static_cast<int>(isBlocked && (spatialRank == 0));
    order.resize(reshapedRank);
    optimizedParams.src_block_dims.resize(reshapedRank);

    size_t spatialShapeSize = 1;
    if (spatialRank != 0) {
        for (int i = batchRank + 1; i < dataRank; i++) {
            spatialShapeSize *= dataDims[i];
        }
    }

    auto decomposeAndTranpose = [&](int axis){
        optimizedParams.src_block_dims[axis] = group;
        optimizedParams.src_block_dims[axis + 1] = groupSize;
        order[axis] = axis + 1;
        order[axis + 1] = axis;
    };

    const int channelDim = 1;
    if (isBlocked) {
        size_t blkSize = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
        size_t CB = div_up(dataDims[1], blkSize);
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        if (axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                order[i] = i;
                optimizedParams.src_block_dims[i] = srcBlockedDims[i];
            }
            decomposeAndTranpose(batchRank);

            order[batchRank + 2] = batchRank + 2;
            optimizedParams.src_block_dims[batchRank + 2] = spatialShapeSize * blkSize;
        } else { // axis on batch
            decomposeAndTranpose(0);
            size_t spatialShapeSize = CB * blkSize;
            for (int i = 2; i < dataRank; i++) {
                spatialShapeSize *= dataDims[i];
            }
            order[2] = 2;
            optimizedParams.src_block_dims[2] = spatialShapeSize;
        }
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        if (axis == channelDim) {  // axis on channel
            order[0] = 0;
            optimizedParams.src_block_dims[0] = dataDims[0];
            order[1] = 1;
            optimizedParams.src_block_dims[1] = spatialShapeSize;
            decomposeAndTranpose(2);
        } else if (axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                if (i == 0) {
                    order[i] = i;
                    optimizedParams.src_block_dims[i] = dataDims[i];
                } else if (i == 1) {
                    order[reshapedRank - 1] = reshapedRank - 1;
                    optimizedParams.src_block_dims[order[reshapedRank - 1]] = dataDims[i];
                } else if (i > 1) {
                    order[i - 1] = i - 1;
                    optimizedParams.src_block_dims[i - 1] = dataDims[i];
                }
            }
            decomposeAndTranpose(batchRank - 1);

            if (spatialRank != 0) {
                order[batchRank + 1] = batchRank + 1;
                optimizedParams.src_block_dims[batchRank + 1] = spatialShapeSize;
            }
        } else { // axis on batch
            decomposeAndTranpose(0);
            order[2] = 2;
            optimizedParams.src_block_dims[2] = spatialShapeSize;
        }
    } else {
        for (int i = 0; i < batchRank; i++) {
            optimizedParams.src_block_dims[i] = dataDims[i];
            order[i] = i;
        }

        decomposeAndTranpose(batchRank);
        if (spatialRank != 0) {
            order[batchRank + 2] = batchRank + 2;
            optimizedParams.src_block_dims[batchRank + 2] = spatialShapeSize;
        }
    }

    prepareOptimizedParams(getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision());
    prepareConfigParams();
}

void MKLDNNShuffleChannelsNode::execute(mkldnn::stream strm) {
    if (permute_kernel) {
        auto srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
        auto dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
        optimizedExecute(srcData, dstData);
    } else {
        auto srcData = reinterpret_cast<const float*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
        auto dstData = reinterpret_cast<float*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
        executeRef(srcData, dstData);
    }
}

void MKLDNNShuffleChannelsNode::executeRef(const float* srcData, float* dstData) {
    size_t batchNum = 1;
    for (int i = 0; i < axis; i++)
        batchNum *= dataDims[i];

    size_t dataLength = 1;
    for (size_t i = axis + 1; i < dataDims.size(); i++)
        dataLength *= dataDims[i];
    if (dataLength == 0)
        THROW_SHCH_ERROR << "has incorrect input parameters dimension.";
    const size_t dataSize = sizeof(float) * dataLength;

    size_t ownDims[CNTR_SIZE] = {batchNum, groupSize, group};
    size_t ownStrides[CNTR_SIZE] = {dataDims[axis], 1, ownDims[1]};
    size_t workAmountDst = ownStrides[0] * ownDims[0];

    auto initter = [&](size_t start, size_t size, size_t* counters) -> size_t {
        size_t i = start;
        size_t idx = 0;
        for (int j = size - 1; j >= 0; j--) {
            counters[j] = i % ownDims[j];
            idx += counters[j] * ownStrides[j];
            i /= ownDims[j];
        }
        return idx;
    };

    auto updater = [&](size_t idx, size_t size, size_t* counters) -> size_t {
        size_t i = 1;
        for (int j = size - 1; j >= 0; j--) {
            counters[j]++;
            if (counters[j] < ownDims[j]) {
                idx += ownStrides[j];
                break;
            } else {
                counters[j] = 0;
                i = 0;
            }
        }
        if (!i) {
            for (idx = 0; i < CNTR_SIZE; ++i)
                idx += counters[i] * ownStrides[i];
        }
        return idx;
    };

    if (dataLength > 1) {
        //  Vectorized & Parallel
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0, srcIdx = 0;
            size_t counters[CNTR_SIZE] = { 0 };
            splitter(workAmountDst, nthr, ithr, start, end);
            srcIdx = initter(start, CNTR_SIZE, counters);
            for (size_t iwork = start, dstIdx = start * dataLength; iwork < end; ++iwork, dstIdx += dataLength) {
                cpu_memcpy(&dstData[dstIdx], &srcData[dataLength * srcIdx], dataSize);
                srcIdx = updater(srcIdx, CNTR_SIZE, counters);
            }
        });
    } else {
        //  Parallel
        parallel_nt(0, [&](const int ithr, const int nthr) {
            size_t start = 0, end = 0, srcIdx = 0;
            size_t counters[CNTR_SIZE] = { 0 };
            splitter(workAmountDst, nthr, ithr, start, end);
            srcIdx = initter(start, CNTR_SIZE, counters);
            for (size_t iwork = start; iwork < end; ++iwork) {
                dstData[iwork] = srcData[srcIdx];
                srcIdx = updater(srcIdx, CNTR_SIZE, counters);
            }
        });
    }
}

bool MKLDNNShuffleChannelsNode::created() const {
    return getType() == ShuffleChannels;
}

REG_MKLDNN_PRIM_FOR(MKLDNNShuffleChannelsNode, ShuffleChannels);
