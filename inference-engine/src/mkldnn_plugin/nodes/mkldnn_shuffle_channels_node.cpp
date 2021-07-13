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

bool MKLDNNShuffleChannelsNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto shuffleChannels = std::dynamic_pointer_cast<const ngraph::op::v0::ShuffleChannels>(op);
        if (!shuffleChannels) {
            errorMessage = "Only opset1 ShuffleChannels operation is supported";
            return false;
        }
        auto shapeSC = shuffleChannels->get_input_shape(0);
        auto rankSC = shapeSC.size();
        auto axisSC = shuffleChannels->get_axis();
        auto groupSC = shuffleChannels->get_group();
        if (axisSC < 0)
            axisSC += rankSC;

        if (axisSC < 0 || axisSC >= rankSC) {
            errorMessage = "gets incorrect axis number, which should be in range of [-inputRank, inputRank).";
            return false;
        }

        if (groupSC == 0 || shapeSC[axisSC] % groupSC) {
            errorMessage = "gets incorrect group parameter('group' must evenly divide the channel dimension).";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNShuffleChannelsNode::MKLDNNShuffleChannelsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), permuteKernel_(nullptr), supportDynamicBatch_(false) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto shuffleChannels = std::dynamic_pointer_cast<const ngraph::op::v0::ShuffleChannels>(op);
    inShape_ = shuffleChannels->get_input_shape(0);
    dataRank_ = inShape_.size();
    axis_ = shuffleChannels->get_axis();
    if (axis_ < 0)
        axis_ += dataRank_;
    group_ = shuffleChannels->get_group();
    groupSize_ = inShape_[axis_] / group_;

    supportDynamicBatch_ = (axis_ != 0);
}

void MKLDNNShuffleChannelsNode::getSupportedDescriptors() {
}

void MKLDNNShuffleChannelsNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8, 16};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
        THROW_SHCH_ERROR << "has unsupported precision: " << precision.name();

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

    // use ncsp as default for non-quantized networks and nspc for quantized
    auto firstCreatorType = isInQuantizedGraph ? TensorDescCreatorTypes::nspc : TensorDescCreatorTypes::ncsp;
    auto secondCreatorType = isInQuantizedGraph ? TensorDescCreatorTypes::ncsp : TensorDescCreatorTypes::nspc;

    addSupportedPrimDesc({{firstCreatorType, precision}},
                         {{firstCreatorType, precision}},
                         impl_type, supportDynamicBatch_);
    addSupportedPrimDesc({{secondCreatorType, precision}},
                         {{secondCreatorType, precision}},
                         impl_type, supportDynamicBatch_);
    // canUseBlocked
    if (axis_ != 1) {
        addSupportedPrimDesc({{TensorDescCreatorTypes::nCsp8c, precision}},
                             {{TensorDescCreatorTypes::nCsp8c, precision}},
                             impl_type, supportDynamicBatch_);
        addSupportedPrimDesc({{TensorDescCreatorTypes::nCsp16c, precision}},
                             {{TensorDescCreatorTypes::nCsp16c, precision}},
                             impl_type, supportDynamicBatch_);
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

    int batchRank = axis_;
    int spatialRank = dataRank_ - axis_ - 1;

    // 2 for decomposed axis dim, 1 for composed spatial dim
    int reshapedRank = batchRank + 2 + static_cast<int>(spatialRank != 0) + static_cast<int>(isBlocked && (spatialRank == 0));
    PermuteParams params;
    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);

    size_t spatialShapeSize = 1;
    if (spatialRank != 0) {
        for (int i = batchRank + 1; i < dataRank_; i++) {
            spatialShapeSize *= inShape_[i];
        }
    }

    auto decomposeAndTranpose = [&](int axis) {
        params.src_block_dims[axis] = group_;
        params.src_block_dims[axis + 1] = groupSize_;
        params.order[axis] = axis + 1;
        params.order[axis + 1] = axis;
    };

    const int channelDim = 1;
    if (isBlocked) {
        size_t blkSize = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims().back();
        size_t CB = div_up(inShape_[1], blkSize);
        SizeVector srcBlockedDims = getParentEdgeAt(0)->getDesc().getBlockingDesc().getBlockDims();
        if (axis_ > channelDim) {  // axis on spatial
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
            for (int i = 2; i < dataRank_; i++) {
                spatialShapeSize *= inShape_[i];
            }
            params.order[2] = 2;
            params.src_block_dims[2] = spatialShapeSize;
        }
    } else if (getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        if (axis_ == channelDim) {  // axis on channel
            params.order[0] = 0;
            params.src_block_dims[0] = inShape_[0];
            params.order[1] = 1;
            params.src_block_dims[1] = spatialShapeSize;
            decomposeAndTranpose(2);
        } else if (axis_ > channelDim) {  // axis on spatial
            for (int i = 0; i < batchRank; i++) {
                if (i == 0) {
                    params.order[i] = i;
                    params.src_block_dims[i] = inShape_[i];
                } else if (i == 1) {
                    params.order[reshapedRank - 1] = reshapedRank - 1;
                    params.src_block_dims[params.order[reshapedRank - 1]] = inShape_[i];
                } else if (i > 1) {
                    params.order[i - 1] = i - 1;
                    params.src_block_dims[i - 1] = inShape_[i];
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
            params.src_block_dims[i] = inShape_[i];
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

    permuteKernel_ = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNShuffleChannelsNode::execute(mkldnn::stream strm) {
    auto srcData = reinterpret_cast<const uint8_t*>(this->getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto dstData = reinterpret_cast<uint8_t*>(this->getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    if (permuteKernel_) {
        if (supportDynamicBatch_)
            permuteKernel_->execute(srcData, dstData, batchToProcess());
        else
            permuteKernel_->execute(srcData, dstData);
    } else {
        THROW_SHCH_ERROR << "does not initialize permute kernel to execute.";
    }
}

bool MKLDNNShuffleChannelsNode::created() const {
    return getType() == ShuffleChannels;
}

REG_MKLDNN_PRIM_FOR(MKLDNNShuffleChannelsNode, ShuffleChannels);
