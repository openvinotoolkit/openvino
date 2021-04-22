// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_shuffle_channels_node.h"

#include <ie_parallel.hpp>
#include <mkldnn_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "common/tensor_desc_creator.h"

#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"

#include <string>
#include <cmath>

#define THROW_SHCH_ERROR IE_THROW() << "ShuffleChannels layer with name '" << getName() << "' "

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu::x64;

MKLDNNShuffleChannelsNode::MKLDNNShuffleChannelsNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), permuteKernel(nullptr), supportDynamicBatch(false) {}

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
    supportDynamicBatch = (axis != 0);

    group = layer->GetParamAsUInt("group", 1);
    if (group == 0 || dataDims[axis] % group)
        THROW_SHCH_ERROR << "gets incorrect group parameter('group' parameter must evenly divide the channel dimension).";

    groupSize = dataDims[axis] / group;
}

void MKLDNNShuffleChannelsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();

    auto srcDims = getParentEdgeAt(0)->getDims();
    int nDims = srcDims.ndims();

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = supportDynamicBatch;
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

    std::vector<TensorDescCreatorTypes> supportedTypes;
    if (nDims > 2) {
        auto canUseBlocked = [=]() {
            return (axis != 1);
        };

        supportedTypes.push_back(TensorDescCreatorTypes::nspc);
        if (canUseBlocked()) {
            if (impl_desc_type::jit_avx512 == impl_type)
                supportedTypes.push_back(TensorDescCreatorTypes::nCsp16c);
            else
                supportedTypes.push_back(TensorDescCreatorTypes::nCsp8c);
        }
    }
    supportedTypes.push_back(TensorDescCreatorTypes::ncsp);
    auto creators = TensorDescCreator::getCommonCreators();
    auto range = TensorDescCreator::makeFilteredRange(creators, nDims, supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].desc = itr->second->createDesc(precision, getParentEdgeAt(0)->getDims().ToSizeVector());
        config.outConfs[0].desc = itr->second->createDesc(precision, getChildEdgeAt(0)->getDims().ToSizeVector());
        supportedPrimitiveDescriptors.emplace_back(config, impl_type, MKLDNNMemoryDesc(config.outConfs.front().desc).getFormat());
    }
}

void MKLDNNShuffleChannelsNode::createPrimitive() {
    if (prim)
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
    PermuteParams params;
    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);
    params.supported_dynamic_batch = supportDynamicBatch;

    size_t spatialShapeSize = 1;
    if (spatialRank != 0) {
        for (int i = batchRank + 1; i < dataRank; i++) {
            spatialShapeSize *= dataDims[i];
        }
    }

    auto decomposeAndTranpose = [&](int axis) {
        params.src_block_dims[axis] = group;
        params.src_block_dims[axis + 1] = groupSize;
        params.order[axis] = axis + 1;
        params.order[axis + 1] = axis;
    };

    const int channelDim = 1;
    if (isBlocked) {
        size_t blkSize = mayiuse(cpu::x64::avx512_common) ? 16 : 8;
        size_t CB = div_up(dataDims[1], blkSize);
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        if (axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                params.order[i] = i;
                params.src_block_dims[i] = srcBlockedDims[i];
            }
            decomposeAndTranpose(batchRank);

            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize * blkSize;
        } else { // axis on batch
            decomposeAndTranpose(0);
            size_t spatialShapeSize = CB * blkSize;
            for (int i = 2; i < dataRank; i++) {
                spatialShapeSize *= dataDims[i];
            }
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        if (axis == channelDim) {  // axis on channel
            params.order[0] = 0;
            params.src_block_dims[0] = dataDims[0];
            params.order[1] = 1;
            params.src_block_dims[1] = spatialShapeSize;
            decomposeAndTranpose(2);
        } else if (axis > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                if (i == 0) {
                    params.order[i] = i;
                    params.src_block_dims[i] = dataDims[i];
                } else if (i == 1) {
                    params.order[reshapedRank - 1] = reshapedRank - 1;
                    params.src_block_dims[params.order[reshapedRank - 1]] = dataDims[i];
                } else if (i > 1) {
                    params.order[i - 1] = i - 1;
                    params.src_block_dims[i - 1] = dataDims[i];
                }
            }
            decomposeAndTranpose(batchRank - 1);

            if (spatialRank != 0) {
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
            params.src_block_dims[i] = dataDims[i];
            params.order[i] = i;
        }

        decomposeAndTranpose(batchRank);
        if (spatialRank != 0) {
            params.order[batchRank + 2] = batchRank + 2;
            params.src_block_dims[batchRank + 2] = spatialShapeSize;
        }
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++)
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNShuffleChannelsNode::execute(mkldnn::stream strm) {
    auto srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    if (permuteKernel) {
        int batch = supportDynamicBatch ? batchToProcess() : dataDims[0];
        permuteKernel->execute(srcData, dstData, batch);
    } else {
        THROW_SHCH_ERROR << "does not initialize permute kernel to execute.";
    }
}

bool MKLDNNShuffleChannelsNode::created() const {
    return getType() == ShuffleChannels;
}

REG_MKLDNN_PRIM_FOR(MKLDNNShuffleChannelsNode, ShuffleChannels);
