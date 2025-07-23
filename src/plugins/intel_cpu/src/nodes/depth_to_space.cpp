// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space.h"

#include <cmath>
#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "common/blocked_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "graph_context.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/permute_kernel.h"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/enum_names.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/util/pp.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl;

namespace ov::intel_cpu::node {

size_t DepthToSpace::DepthToSpaceAttrs::hash() const {
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

    return seed;
}

bool DepthToSpace::DepthToSpaceAttrs::operator==(const DepthToSpaceAttrs& rhs) const {
    bool result = layoutType == rhs.layoutType && mode == rhs.mode && blockSize == rhs.blockSize &&
                  blockStep == rhs.blockStep && dataSize == rhs.dataSize && nSpatialDims == rhs.nSpatialDims &&
                  srcBlockedDims == rhs.srcBlockedDims;

    return result;
}

bool DepthToSpace::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto depthToSpace = ov::as_type_ptr<const ov::op::v0::DepthToSpace>(op);
        if (!depthToSpace) {
            errorMessage = "Only v0 DepthToSpace operation is supported";
            return false;
        }
        const auto mode = depthToSpace->get_mode();
        if (none_of(mode,
                    ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                    ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST)) {
            errorMessage = "Does not support mode: " + ov::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

DepthToSpace::DepthToSpace(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    CPU_NODE_ASSERT(all_of(1U, inputShapes.size(), outputShapes.size()), "has incorrect number of input/output edges!");

    auto depthToSpace = ov::as_type_ptr<const ov::op::v0::DepthToSpace>(op);
    CPU_NODE_ASSERT(depthToSpace, "supports only v0");

    const auto modeNgraph = depthToSpace->get_mode();
    if (modeNgraph == ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST) {
        attrs.mode = Mode::BLOCKS_FIRST;
    } else if (modeNgraph == ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST) {
        attrs.mode = Mode::DEPTH_FIRST;
    } else {
        CPU_NODE_THROW("doesn't support mode: ", ov::as_string(modeNgraph));
    }

    attrs.blockSize = depthToSpace->get_block_size();
    CPU_NODE_ASSERT(attrs.blockSize != 0, "has incorrect block_size parameter is zero!");

    const size_t srcRank = getInputShapeAtPort(0).getRank();
    const size_t dstRank = getOutputShapeAtPort(0).getRank();

    CPU_NODE_ASSERT(srcRank >= 3, "has incorrect number of input dimensions");
    CPU_NODE_ASSERT(srcRank <= 5, "doesn't support dimensions with rank greater than 5");
    CPU_NODE_ASSERT(srcRank == dstRank, "has incorrect number of input/output dimensions");

    const size_t nSpatialDims = srcRank - 2;
    attrs.blockStep = static_cast<size_t>(std::pow(attrs.blockSize, nSpatialDims));
}

void DepthToSpace::getSupportedDescriptors() {}

void DepthToSpace::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);

    impl_desc_type impl_type = impl_desc_type::ref;
    if (cpu::x64::mayiuse(cpu::x64::avx512_core)) {
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
                   (srcDims[1] / block) % attrs.blockStep == 0 &&
                   (attrs.mode == Mode::DEPTH_FIRST ? block % attrs.blockStep == 0 : true);
        };

        supportedTypes.push_back(LayoutType::nspc);
        if (canUseBlocked(8LU)) {
            supportedTypes.push_back(LayoutType::nCsp8c);
        }
        if (canUseBlocked(16LU)) {
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

void DepthToSpace::createPrimitive() {
    auto dstMemPtr = getDstMemoryAtPort(0);
    auto srcMemPtr = getSrcMemoryAtPort(0);
    CPU_NODE_ASSERT(dstMemPtr, "has null destination memory");
    CPU_NODE_ASSERT(srcMemPtr, "has null input memory");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "has unidentified preferable primitive descriptor");

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.dataSize = memoryDesc.getPrecision().size();
    attrs.nSpatialDims = memoryDesc.getShape().getRank() - 2;
    if (memoryDesc.hasLayoutType(LayoutType::nCsp16c)) {
        attrs.layoutType = LayoutType::nCsp16c;
    } else if (memoryDesc.hasLayoutType(LayoutType::nCsp8c)) {
        attrs.layoutType = LayoutType::nCsp8c;
    } else if (memoryDesc.hasLayoutType(LayoutType::nspc)) {
        attrs.layoutType = LayoutType::nspc;
    } else {
        attrs.layoutType = LayoutType::ncsp;
    }

    if (inputShapesDefined()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void DepthToSpace::prepareParams() {
    attrs.srcBlockedDims = getSrcMemoryAtPort(0)->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto builder = [](const DepthToSpaceAttrs& key) -> std::shared_ptr<DepthToSpaceExecutor> {
        return std::make_shared<DepthToSpaceExecutor>(key);
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(attrs, builder);
    CPU_NODE_ASSERT(result.first, "DepthToSpaceExecutor was not found.");

    execPtr = result.first;
}

DepthToSpace::DepthToSpaceExecutor::DepthToSpaceExecutor(const DepthToSpaceAttrs& attrs) {
    OPENVINO_ASSERT(
        any_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c, LayoutType::nspc, LayoutType::ncsp),
        "DepthToSpace executor supports only 'nCsp16c', 'nCsp8c', 'nspc' or 'ncsp' layouts.");

    const bool isBlocked = any_of(attrs.layoutType, LayoutType::nCsp16c, LayoutType::nCsp8c);
    const bool isChannelsFirst = attrs.layoutType == LayoutType::nspc;
    const size_t nDims = attrs.srcBlockedDims.size();
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
    params.src_block_dims[0] = attrs.srcBlockedDims[0];

    // reshaping of src dimensions and creating the permutation order for each layout:
    // new shape: mode = blocks_first [N, block_size, block_size, ..., block_size, C / (block_size ^ K), D1, D2, ...,
    // DK]
    //            mode = depth_first  [N, C / (block_size ^ K), block_size, block_size, ..., block_size, D1, D2, ...,
    //            DK]
    // order    : mode = blocks_first : [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ..., K + (K + 1), K]
    //            mode = depth_first  : [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1]
    // where `k` is number of spatial dimensions

    auto reshapeAndSetPermOrder =
        [&](const size_t idx1, const size_t idx2, const size_t shift, const VectorDims& dims) {
            for (size_t i = 0; i < attrs.nSpatialDims; i++) {
                params.order[i * 2 + shift] = i + idx1;
                params.order[i * 2 + shift + 1] = i + idx2;

                params.src_block_dims[params.order[i * 2 + shift]] = dims[i + shift];
                params.src_block_dims[params.order[i * 2 + shift + 1]] = attrs.blockSize;
            }
        };

    if (isBlocked) {
        size_t orderShiftForBlocks = 0;
        size_t orderShiftForDims = 0;
        if (attrs.mode == Mode::BLOCKS_FIRST) {
            orderShiftForBlocks = 1;
            orderShiftForDims = attrs.nSpatialDims + 2;

            params.src_block_dims[attrs.nSpatialDims + 1] = attrs.srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[lastIdx] = attrs.srcBlockedDims.back();

            params.order[1] = attrs.nSpatialDims + 1;
            params.order[lastIdx] = lastIdx;
        } else {
            orderShiftForBlocks = attrs.nSpatialDims + 4;
            orderShiftForDims = 3;

            size_t newBlockSize = attrs.srcBlockedDims.back() / attrs.blockStep;
            size_t newBlocksCount = attrs.srcBlockedDims[1] / attrs.blockStep;
            params.src_block_dims[1] = newBlocksCount;
            params.src_block_dims[2] = attrs.srcBlockedDims[1] / newBlocksCount;
            params.src_block_dims[lastIdx - attrs.nSpatialDims] = newBlockSize;

            params.order[1] = 1;
            params.order[2] = 3;
            params.order[lastIdx - 1] = 2;
            params.order[lastIdx] = lastIdx - attrs.nSpatialDims;
        }

        reshapeAndSetPermOrder(orderShiftForDims, orderShiftForBlocks, firstSpatialOrder, attrs.srcBlockedDims);
    } else if (isChannelsFirst) {
        firstSpatialOrder = 1;

        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + attrs.nSpatialDims + 1;
        params.order[lastIdx] = attrs.mode == Mode::DEPTH_FIRST ? attrs.nSpatialDims + 1 : lastIdx;
        params.src_block_dims[params.order[lastIdx]] = attrs.srcBlockedDims.back() / attrs.blockStep;

        reshapeAndSetPermOrder(firstSpatialOrder, shift, firstSpatialOrder, attrs.srcBlockedDims);
    } else {
        size_t shift = static_cast<size_t>(attrs.mode == DEPTH_FIRST) + 1;
        params.order[1] = attrs.mode == DEPTH_FIRST ? 1 : attrs.nSpatialDims + 1;
        params.src_block_dims[params.order[1]] = attrs.srcBlockedDims[1] / attrs.blockStep;

        reshapeAndSetPermOrder(attrs.nSpatialDims + firstSpatialOrder, shift, firstSpatialOrder, attrs.srcBlockedDims);
    }

    std::iota(params.src_block_order.begin(), params.src_block_order.end(), 0);
    std::iota(params.dst_block_order.begin(), params.dst_block_order.end(), 0);
    for (size_t i = 0; i < reshapedRank; i++) {
        params.dst_block_dims[i] = params.src_block_dims[params.order[i]];
    }

    permuteKernel = std::make_unique<PermuteKernel>(params);
}

void DepthToSpace::DepthToSpaceExecutor::exec(const MemoryPtr& srcMemPtr, const MemoryPtr& dstMemPtr, int MB) {
    OPENVINO_ASSERT(permuteKernel, "Could not execute. Kernel for Transpose node was not compiled.");

    const auto* srcData = srcMemPtr->getDataAs<const uint8_t>();
    auto* dstData = dstMemPtr->getDataAs<uint8_t>();

    permuteKernel->execute(srcData, dstData, MB);
}

void DepthToSpace::execute([[maybe_unused]] const dnnl::stream& strm) {
    CPU_NODE_ASSERT(execPtr, "doesn't have a compiled executor.");

    int MB = getSrcMemoryAtPort(0)->getStaticDims()[0];
    execPtr->exec(getSrcMemoryAtPort(0), getDstMemoryAtPort(0), MB);
}

void DepthToSpace::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool DepthToSpace::created() const {
    return getType() == Type::DepthToSpace;
}

}  // namespace ov::intel_cpu::node
