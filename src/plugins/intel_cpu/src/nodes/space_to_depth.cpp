// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "space_to_depth.h"

#include <dnnl_extension_utils.h>
#include <utils/general_utils.h>

#include <cmath>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <string>

#include "common/blocked_desc_creator.h"

#define THROW_ERROR IE_THROW() << "SpaceToDepth layer with name '" << getName() << "' "

using namespace InferenceEngine;
using namespace dnnl;
using namespace dnnl::impl;

namespace ov {
namespace intel_cpu {
namespace node {

bool SpaceToDepth::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op,
                                                  std::string& errorMessage) noexcept {
    try {
        const auto spaceToDepth = ov::as_type_ptr<const ngraph::opset1::SpaceToDepth>(op);
        if (!spaceToDepth) {
            errorMessage = "Only opset1 SpaceToDepth operation is supported";
            return false;
        }
        const auto mode = spaceToDepth->get_mode();
        if (!one_of(mode,
                    ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
                    ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST)) {
            errorMessage = "Does not support mode: " + ngraph::as_string(mode);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

SpaceToDepth::SpaceToDepth(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        THROW_ERROR << "has incorrect number of input/output edges!";

    auto spaceToDepth = ov::as_type_ptr<const ngraph::opset1::SpaceToDepth>(op);
    if (!spaceToDepth)
        THROW_ERROR << "supports only opset1";

    const auto modeNgraph = spaceToDepth->get_mode();
    if (modeNgraph == ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST) {
        attrs.mode = SpaceToDepthAttrs::Mode::BLOCKS_FIRST;
    } else if (modeNgraph == ngraph::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST) {
        attrs.mode = SpaceToDepthAttrs::Mode::DEPTH_FIRST;
    } else {
        THROW_ERROR << "doesn't support mode: " << ngraph::as_string(modeNgraph);
    }

    attrs.blockSize = spaceToDepth->get_block_size();
    if (attrs.blockSize == 0)
        THROW_ERROR << "has incorrect block_size parameter is zero!";

    const size_t srcRank = getInputShapeAtPort(0).getRank();
    const size_t dstRank = getOutputShapeAtPort(0).getRank();
    if (srcRank < 3)
        THROW_ERROR << "has incorrect number of input dimensions";
    if (srcRank > 5)
        THROW_ERROR << "doesn't support dimensions with rank greater than 5";
    if (srcRank != dstRank)
        THROW_ERROR << "has incorrect number of input/output dimensions";
    attrs.nSpatialDims = srcRank - 2;
    attrs.blockStep = static_cast<size_t>(std::pow(attrs.blockSize, attrs.nSpatialDims));
}

void SpaceToDepth::getSupportedDescriptors() {}

void SpaceToDepth::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);

    attrs.implDescType = impl_desc_type::ref;
    if (cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        attrs.implDescType = impl_desc_type::jit_avx512;
    } else if (cpu::x64::mayiuse(cpu::x64::avx2)) {
        attrs.implDescType = impl_desc_type::jit_avx2;
    } else if (cpu::x64::mayiuse(cpu::x64::sse41)) {
        attrs.implDescType = impl_desc_type::jit_sse42;
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
        auto canUseBlocked = [=](const size_t block) {
            return srcDims[1] != Shape::UNDEFINED_DIM && srcDims[1] % block == 0 &&
                   (attrs.mode == SpaceToDepthAttrs::Mode::DEPTH_FIRST ? block % attrs.blockStep == 0 : true);
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

    auto supportedPrimitiveDescriptorsBuilder = [this](NodeConfig config, impl_desc_type implDescType) {
        std::vector<MemoryDescPtr> srcMemoryDescs = {config.inConfs[0].getMemDesc()}, dstMemoryDescs = {config.outConfs[0].getMemDesc()};
        auto factory = std::make_shared<SpaceToDepthExecutorFactory>(attrs, srcMemoryDescs, dstMemoryDescs,
                                                                     std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
        supportedPrimitiveDescriptors.emplace_back(config, implDescType, factory);
    };

    for (auto itr = range.first; itr != range.second; ++itr) {
        config.inConfs[0].setMemDesc(itr->second->createSharedDesc(precision, inputDataShape));
        config.outConfs[0].setMemDesc(itr->second->createSharedDesc(precision, outputDataShape));
        supportedPrimitiveDescriptorsBuilder(config, attrs.implDescType);
    }
}

void SpaceToDepth::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR << "has unidentified preferable primitive descriptor";

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.dataSize = memoryDesc.getPrecision().size();
    attrs.layoutType = memoryDesc.hasLayoutType(LayoutType::nCsp16c)
                           ? LayoutType::nCsp16c
                           : memoryDesc.hasLayoutType(LayoutType::nCsp8c)
                                 ? LayoutType::nCsp8c
                                 : memoryDesc.hasLayoutType(LayoutType::nspc) ? LayoutType::nspc : LayoutType::ncsp;

    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void SpaceToDepth::prepareParams() {
    attrs.srcBlockedDims =
        getParentEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    attrs.destBlockedDims =
        getChildEdgeAt(0)->getMemoryPtr()->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto builder = [this](const SpaceToDepthAttrs& key) -> std::shared_ptr<SpaceToDepthExecutor> {
        dnnl::primitive_attr attr;
        auto selectedPD = getSelectedPrimitiveDescriptor();
        std::vector<MemoryDescPtr> srcDescs = {getParentEdgeAt(0)->getMemoryPtr()->getDescPtr()};
        std::vector<MemoryDescPtr> dstDescs = {getChildEdgeAt(0)->getMemoryPtr()->getDescPtr()};
        auto spaceToDepthExecutor = selectedPD->getExecutorFactoryAs<SpaceToDepthExecutorFactory>()->makeExecutor(attrs, srcDescs, dstDescs, attr);
        selectedPD->setImplementationType(spaceToDepthExecutor->getImplType());
        return spaceToDepthExecutor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(attrs, builder);
    if (!result.first) {
        IE_THROW() << "SpaceToDepthExecutor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

void SpaceToDepth::execute(dnnl::stream strm) {
    if (!execPtr) {
        THROW_ERROR << "doesn't have a compiled executor.";
    }
    const int MB = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims()[0];
    execPtr->exec({getParentEdgeAt(0)->getMemoryPtr()}, {getChildEdgeAt(0)->getMemoryPtr()}, MB);
}

void SpaceToDepth::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool SpaceToDepth::created() const {
    return getType() == Type::SpaceToDepth;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
