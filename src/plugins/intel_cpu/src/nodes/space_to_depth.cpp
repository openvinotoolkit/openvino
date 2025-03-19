// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth.h"

#include <cmath>
#include <memory>
#include <string>

#include "common/blocked_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "openvino/opsets/opset1.hpp"
#include "openvino/util/pp.hpp"
#include "utils/general_utils.h"

using namespace dnnl;
using namespace dnnl::impl;

namespace ov::intel_cpu::node {

size_t SpaceToDepth::SpaceToDepthAttrs::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, layoutType);
    seed = hash_combine(seed, mode);
    seed = hash_combine(seed, blockSize);
    seed = hash_combine(seed, blockStep);
    seed = hash_combine(seed, dataSize);
    seed = hash_combine(seed, nSpatialDims);
    seed = get_vector_hash(seed, srcBlockedDims);
    seed = get_vector_hash(seed, destBlockedDims);

    return seed;
}

bool SpaceToDepth::SpaceToDepthAttrs::operator==(const SpaceToDepthAttrs& rhs) const {
    bool result = layoutType == rhs.layoutType && mode == rhs.mode && blockSize == rhs.blockSize &&
                  blockStep == rhs.blockStep && dataSize == rhs.dataSize && nSpatialDims == rhs.nSpatialDims &&
                  srcBlockedDims == rhs.srcBlockedDims && destBlockedDims == rhs.destBlockedDims;

    return result;
}

bool SpaceToDepth::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto spaceToDepth = ov::as_type_ptr<const ov::opset1::SpaceToDepth>(op);
        if (!spaceToDepth) {
            errorMessage = "Only opset1 SpaceToDepth operation is supported";
            return false;
        }
        const auto mode = spaceToDepth->get_mode();
        if (!one_of(mode,
                    ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                    ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST)) {
            errorMessage = "Does not support mode: " + ov::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SpaceToDepth::SpaceToDepth(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    if (inputShapes.size() != 1 || outputShapes.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    auto spaceToDepth = ov::as_type_ptr<const ov::opset1::SpaceToDepth>(op);
    if (!spaceToDepth) {
        THROW_CPU_NODE_ERR("supports only opset1");
    }

    const auto modeNgraph = spaceToDepth->get_mode();
    if (modeNgraph == ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST) {
        attrs.mode = Mode::BLOCKS_FIRST;
    } else if (modeNgraph == ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST) {
        attrs.mode = Mode::DEPTH_FIRST;
    } else {
        THROW_CPU_NODE_ERR("doesn't support mode: ", ov::as_string(modeNgraph));
    }

    attrs.blockSize = spaceToDepth->get_block_size();
    if (attrs.blockSize == 0) {
        THROW_CPU_NODE_ERR("has incorrect block_size parameter is zero!");
    }

    const size_t srcRank = getInputShapeAtPort(0).getRank();
    const size_t dstRank = getOutputShapeAtPort(0).getRank();
    if (srcRank < 3) {
        THROW_CPU_NODE_ERR("has incorrect number of input dimensions");
    }
    if (srcRank > 5) {
        THROW_CPU_NODE_ERR("doesn't support dimensions with rank greater than 5");
    }
    if (srcRank != dstRank) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output dimensions");
    }
    attrs.nSpatialDims = srcRank - 2;
    attrs.blockStep = static_cast<size_t>(std::pow(attrs.blockSize, attrs.nSpatialDims));
}

void SpaceToDepth::getSupportedDescriptors() {}

void SpaceToDepth::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    impl_desc_type impl_type = impl_desc_type::ref;
    if (cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (cpu::x64::mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    }

    NodeConfig config;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace(-1);
    config.inConfs[0].constant(false);
    config.outConfs[0].inPlace(-1);
    config.outConfs[0].constant(false);

    const auto& inputDataShape = getInputShapeAtPort(0);
    const auto& outputDataShape = getOutputShapeAtPort(0);

    std::vector<LayoutType> supportedTypes;
    if (inputDataShape.getRank() > 2) {
        const auto& srcDims = inputDataShape.getDims();
        auto canUseBlocked = [OV_CAPTURE_CPY_AND_THIS](const size_t block) {
            return srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % block == 0 &&
                   (attrs.mode == Mode::DEPTH_FIRST ? block % attrs.blockStep == 0 : true);
        };

        supportedTypes.push_back(LayoutType::nspc);
        if (canUseBlocked(8lu)) {
            supportedTypes.push_back(LayoutType::nCsp8c);
        }
        if (canUseBlocked(16lu)) {
            supportedTypes.push_back(LayoutType::nCsp16c);
        }
    }
    supportedTypes.push_back(LayoutType::ncsp);
    auto creators = BlockedDescCreator::getCommonCreators();
    auto range = BlockedDescCreator::makeFilteredRange(creators, inputDataShape.getRank(), supportedTypes);

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].setMemDesc(itr->second->createSharedDesc(precision, inputDataShape));
        config.outConfs[0].setMemDesc(itr->second->createSharedDesc(precision, outputDataShape));
        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    }
}

void SpaceToDepth::createPrimitive() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    if (!dstMemPtr) {
        THROW_CPU_NODE_ERR("has null destination memory");
    }
    if (!srcMemPtr) {
        THROW_CPU_NODE_ERR("has null input memory");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has unidentified preferable primitive descriptor");
    }

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.dataSize = memoryDesc.getPrecision().size();
    attrs.layoutType = memoryDesc.hasLayoutType(LayoutType::nCsp16c)  ? LayoutType::nCsp16c
                       : memoryDesc.hasLayoutType(LayoutType::nCsp8c) ? LayoutType::nCsp8c
                       : memoryDesc.hasLayoutType(LayoutType::nspc)   ? LayoutType::nspc
                                                                      : LayoutType::ncsp;

    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void SpaceToDepth::prepareParams() {
    attrs.srcBlockedDims = getSrcMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    attrs.destBlockedDims = getDstMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto builder = [](const SpaceToDepthAttrs& key) -> std::shared_ptr<SpaceToDepthExecutor> {
        return std::make_shared<SpaceToDepthExecutor>(key);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(attrs, builder);
    if (!result.first) {
        THROW_CPU_NODE_ERR("executor was not found.");
    }

    execPtr = result.first;
}

SpaceToDepth::SpaceToDepthExecutor::SpaceToDepthExecutor(const SpaceToDepthAttrs& attrs) {
    if (!one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c, LayoutType::nspc, LayoutType::ncsp)) {
        OPENVINO_THROW("SpaceToDepth executor supports only 'nCsp16c', 'nCsp8c', "
                       "'nspc' or 'ncsp' layouts.");
    }

    const bool isBlocked = one_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsFirst = attrs.layoutType == LayoutType::nspc;
    const auto& srcBlockedDims = attrs.srcBlockedDims;
    const auto& dstBlockedDims = attrs.destBlockedDims;

    size_t nDims = srcBlockedDims.size();

    const size_t reshapedRank =
        nDims + attrs.nSpatialDims + static_cast<int>(isBlocked && attrs.mode == Mode::DEPTH_FIRST);
    const size_t lastIdx = reshapedRank - 1;
    size_t firstSpatialOrder = 2;

    PermuteParams params;
    params.data_size = attrs.dataSize;
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
        [&](const size_t idx1, const size_t idx2, const size_t shift, const VectorDims& dims) {
            for (size_t i = 0; i < attrs.nSpatialDims; i++) {
                params.order[i + idx1] = i * 2 + shift;
                params.order[i + idx2] = i * 2 + shift + 1;

                params.src_block_dims[params.order[i + idx1]] = dims[i + shift];
                params.src_block_dims[params.order[i + idx2]] = attrs.blockSize;
            }
        };

    if (isBlocked) {
        size_t orderShiftForBlocks, orderShiftForDims;
        if (attrs.mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = attrs.nSpatialDims + 2;
            orderShiftForDims = 1;

            params.order[attrs.nSpatialDims + 1] = 1;
            params.order[lastIdx] = lastIdx;

            params.src_block_dims[params.order[attrs.nSpatialDims + 1]] = srcBlockedDims[1];
            params.src_block_dims[params.order[lastIdx]] = srcBlockedDims.back();
        } else {
            orderShiftForBlocks = 3;
            orderShiftForDims = attrs.nSpatialDims + 4;

            size_t extraBlockSize = srcBlockedDims.back() / attrs.blockStep;
            params.src_block_dims[1] = srcBlockedDims[1];
            params.src_block_dims[lastIdx] = extraBlockSize;
            params.src_block_dims[lastIdx - 1] = attrs.blockStep;

            params.order[1] = 1;
            params.order[2] = lastIdx - 1;
            params.order[lastIdx - attrs.nSpatialDims] = lastIdx;
        }

        reshapeAndSetPermOrder(orderShiftForBlocks, orderShiftForDims, firstSpatialOrder, dstBlockedDims);
    } else if (isChannelsFirst) {
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + attrs.nSpatialDims + 1;
        params.order[attrs.mode == Mode::DEPTH_FIRST ? attrs.nSpatialDims + 1 : lastIdx] = lastIdx;
        params.src_block_dims[lastIdx] = srcBlockedDims.back();

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, dstBlockedDims);
    } else {
        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + 1;
        params.order[attrs.mode == Mode::DEPTH_FIRST ? 1 : attrs.nSpatialDims + 1] = 1;
        params.src_block_dims[1] = srcBlockedDims[1];

        reshapeAndSetPermOrder(attrs.nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, dstBlockedDims);
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++) {
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];
    }

    permuteKernel = std::make_unique<PermuteKernel>(params);
}

void SpaceToDepth::SpaceToDepthExecutor::exec(const uint8_t* srcData, uint8_t* dstData, const int MB) {
    if (!permuteKernel) {
        OPENVINO_THROW("Could not execute. Kernel for Transpose node was not compiled.");
    }
    permuteKernel->execute(srcData, dstData, MB);
}

void SpaceToDepth::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("doesn't have a compiled executor.");
    }
    const auto* srcData = getSrcDataAtPortAs<const uint8_t>(0);
    auto* dstData = getDstDataAtPortAs<uint8_t>(0);
    const int MB = getSrcMemoryAtPort(0)->getStaticDims()[0];
    execPtr->exec(srcData, dstData, MB);
}

void SpaceToDepth::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool SpaceToDepth::created() const {
    return getType() == Type::SpaceToDepth;
}

}  // namespace ov::intel_cpu::node
