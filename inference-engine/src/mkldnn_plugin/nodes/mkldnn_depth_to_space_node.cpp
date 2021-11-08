// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depth_to_space_node.h"

#include <cpu/x64/jit_generator.hpp>
#include <mkldnn_extension_utils.h>
#include "common/blocked_desc_creator.h"
#include <utils/general_utils.h>
#include <ngraph/opsets/opset1.hpp>

#include <string>
#include <cmath>

#define THROW_ERROR IE_THROW() << "DepthToSpace layer with name '" << getName() << "' "

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn::impl;

bool MKLDNNDepthToSpaceNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto depthToSpace = ov::as_type_ptr<const ngraph::opset1::DepthToSpace>(op);
        if (!depthToSpace) {
            errorMessage = "Only opset1 DepthToSpace operation is supported";
            return false;
        }
        const auto mode = depthToSpace->get_mode();
        if (!one_of(mode, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST)) {
            errorMessage = "Does not support mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNDepthToSpaceNode::MKLDNNDepthToSpaceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges!";

    const auto depthToSpace = ov::as_type_ptr<const ngraph::opset1::DepthToSpace>(op);

    const auto modeNgraph = depthToSpace->get_mode();
    if (modeNgraph == ngraph::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) {
        attrs.mode = Mode::BLOCKS_FIRST;
    } else if (modeNgraph == ngraph::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST) {
        attrs.mode = Mode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << ngraph::as_string(modeNgraph);
    }

    attrs.blockSize = depthToSpace->get_block_size();
    if (attrs.blockSize == 0)
        THROW_ERROR << "has incorrect block_size parameter is zero!";

    const int srcRank = getInputShapeAtPort(0).getRank();
    const int dstRank = getOutputShapeAtPort(0).getRank();

    if (srcRank < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcRank > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";
    if (srcRank != dstRank)
        THROW_ERROR << "has incorrect number of input/output dimensions";

    size_t nSpatialDims = srcRank - 2;
    attrs.blockStep = static_cast<size_t>(std::pow(attrs.blockSize, nSpatialDims));
}

void MKLDNNDepthToSpaceNode::getSupportedDescriptors() {}

void MKLDNNDepthToSpaceNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);

    impl_desc_type impl_type;
    if (cpu::x64::mayiuse(cpu::x64::avx512_common)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (cpu::x64::mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    NodeConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;

    const auto& inputDataShape = getInputShapeAtPort(0);
    const auto& outputDataShape = getOutputShapeAtPort(0);

    std::vector<LayoutType> supportedTypes;
    if (inputDataShape.getRank() > 2) {
        const auto& srcDims = inputDataShape.getDims();
        auto canUseBlocked = [=](const size_t block) {
            return srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % block == 0 && (srcDims[1] / block) % attrs.blockStep == 0 &&
                   (attrs.mode == Mode::DEPTH_FIRST ? block % attrs.blockStep == 0 : true);
        };

        supportedTypes.push_back(LayoutType::nspc);
        if (canUseBlocked(8lu))
            supportedTypes.push_back(LayoutType::nCsp8c);
        if (canUseBlocked(16lu))
            supportedTypes.push_back(LayoutType::nCsp16c);
    }
    supportedTypes.push_back(LayoutType::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, inputDataShape.getRank(), supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].desc = itr->second->createSharedDesc(precision, inputDataShape);
        config.outConfs[0].desc = itr->second->createSharedDesc(precision, outputDataShape);
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    }
}

void MKLDNNDepthToSpaceNode::createPrimitive() {
    dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";
    attrs.dataSize = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision().size();

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void MKLDNNDepthToSpaceNode::prepareParams() {
    execPtr = std::make_shared<DepthToSpaceExecutor>(attrs, getParentEdgeAt(0)->getMemoryPtr(), getChildEdgeAt(0)->getMemoryPtr());
}

MKLDNNDepthToSpaceNode::DepthToSpaceExecutor::DepthToSpaceExecutor(const DepthToSpaceAttrs& attrs,
                                                                   const MKLDNNMemoryPtr& srcMemPtr,
                                                                   const MKLDNNMemoryPtr& dstMemPtr) {
    VectorDims srcDims = srcMemPtr->getStaticDims();
    VectorDims dstDims = dstMemPtr->getStaticDims();
    size_t nDims = srcDims.size();
    const size_t nSpatialDims = nDims - 2;

    const bool isBlocked = srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp8c) ||
                           srcMemPtr->getDesc().hasLayoutType(LayoutType::nCsp16c);
    const size_t reshapedRank = nDims + nSpatialDims + static_cast<int>(isBlocked) + static_cast<int>(isBlocked && attrs.mode == Mode::DEPTH_FIRST);
    const size_t lastIdx = reshapedRank - 1;
    size_t firstSpatialOrder = 2;

    PermuteParams params;
    params.data_size = attrs.dataSize;
    params.order.resize(reshapedRank, 0);
    params.src_block_order.resize(reshapedRank);
    params.dst_block_order.resize(reshapedRank);
    params.dst_block_dims.resize(reshapedRank);
    params.src_block_dims.resize(reshapedRank);
    params.src_block_dims[0] = srcDims[0];

    // reshaping of src dimensions and creating the permutation order for each layout:
    // new shape: mode = blocks_first [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ..., DK]
    //            mode = depth_first  [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ..., DK]
    // order    : mode = blocks_first : [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K]
    //            mode = depth_first  : [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1]
    // where `k` is number of spatial dimensions

    auto reshapeAndSetPermOrder = [&](const size_t idx1, const size_t idx2, const size_t shift, const SizeVector& dims) {
        for (size_t i = 0; i < nSpatialDims; i++) {
            params.order[i * 2 + shift] = i + idx1;
            params.order[i * 2 + shift + 1] = i + idx2;

            params.src_block_dims[params.order[i * 2 + shift]] = dims[i + shift];
            params.src_block_dims[params.order[i * 2 + shift + 1]] = attrs.blockSize;
        }
    };

    if (isBlocked) {
        VectorDims srcBlockedDims = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();

        size_t orderShiftForBlocks, orderShiftForDims;
        if (attrs.mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = 1;
            orderShiftForDims = nSpatialDims + 2;

            params.src_block_dims[nSpatialDims + 1] = srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[lastIdx] = srcBlockedDims.back();

            params.order[1] = nSpatialDims + 1;
            params.order[lastIdx] = lastIdx;
        } else {
            orderShiftForBlocks = nSpatialDims + 4;
            orderShiftForDims = 3;

            size_t newBlockSize = srcBlockedDims.back() / attrs.blockStep;
            size_t newBlocksCount = srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[1] = newBlocksCount;
            params.src_block_dims[2] = srcBlockedDims[1] / newBlocksCount;
            params.src_block_dims[lastIdx - nSpatialDims] = newBlockSize;

            params.order[1] = 1;
            params.order[2] = 3;
            params.order[lastIdx - 1] = 2;
            params.order[lastIdx] = lastIdx - nSpatialDims;
        }

        reshapeAndSetPermOrder(orderShiftForDims, orderShiftForBlocks, firstSpatialOrder, srcBlockedDims);
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
        srcDims.push_back(srcDims[1]);
        srcDims.erase(srcDims.begin() + 1);
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + nSpatialDims + 1;
        params.order[lastIdx] = attrs.mode == Mode::DEPTH_FIRST ? nSpatialDims + 1 : lastIdx;
        params.src_block_dims[params.order[lastIdx]] = srcDims.back() / attrs.blockStep;

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, srcDims);
    } else {
        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + 1;
        params.order[1] = attrs.mode == DEPTH_FIRST ? 1 : nSpatialDims + 1;
        params.src_block_dims[params.order[1]] = srcDims[1] / attrs.blockStep;

        reshapeAndSetPermOrder(nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, srcDims);
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++)
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

void MKLDNNDepthToSpaceNode::DepthToSpaceExecutor::exec(MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr, const int MB) {
    if (!permuteKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());

    permuteKernel->execute(srcData, dstData, MB);
}

void MKLDNNDepthToSpaceNode::execute(mkldnn::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Could not execute Transpose node. Primitive was not created.";
    }

    int MB = isDynamicNode() ? srcMemPtr->getStaticDims()[0] : batchToProcess();
    execPtr->exec(srcMemPtr, dstMemPtr, MB);
}

void MKLDNNDepthToSpaceNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

bool MKLDNNDepthToSpaceNode::created() const {
    return getType() == DepthToSpace;
}
REG_MKLDNN_PRIM_FOR(MKLDNNDepthToSpaceNode, DepthToSpace);
