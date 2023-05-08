// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shuffle_channels.h"

#include <ie_parallel.hpp>
#include <dnnl_extension_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "common/blocked_desc_creator.h"

#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"

#include <string>
#include <cmath>
#include <common/primitive_hashing_utils.hpp>

#define THROW_SHCH_ERROR IE_THROW() << "ShuffleChannels layer with name '" << getName() << "' "

using namespace dnnl;
using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

bool ShuffleChannels::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto shuffleChannels = ov::as_type_ptr<const ngraph::op::v0::ShuffleChannels>(op);
        if (!shuffleChannels) {
            errorMessage = "Only opset1 ShuffleChannels operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ShuffleChannels::ShuffleChannels(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    if (inputShapes.size() != 1 || outputShapes.size() != 1)
        THROW_SHCH_ERROR << "has incorrect number of input/output edges.";

    auto shuffleChannels = ov::as_type_ptr<const ngraph::op::v0::ShuffleChannels>(op);
    attrs.group = shuffleChannels->get_group();
    attrs.axis = shuffleChannels->get_axis();
    attrs.dataRank = getInputShapeAtPort(0).getRank();
    if (attrs.axis < 0)
        attrs.axis += attrs.dataRank;
}

void ShuffleChannels::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    const std::set<size_t> supported_precision_sizes = {1, 2, 4, 8, 16};
    if (supported_precision_sizes.find(precision.size()) == supported_precision_sizes.end())
        THROW_SHCH_ERROR << "has unsupported precision: " << precision.name();

    if (mayiuse(cpu::x64::avx512_core)) {
        attrs.implDescType = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        attrs.implDescType = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        attrs.implDescType = impl_desc_type::jit_sse42;
    } else {
        attrs.implDescType = impl_desc_type::ref;
    }

    // use ncsp as default for non-quantized networks and nspc for quantized
    auto firstCreatorType = context->isGraphQuantized() ? LayoutType::nspc : LayoutType::ncsp;
    auto secondCreatorType = context->isGraphQuantized() ? LayoutType::ncsp : LayoutType::nspc;

    addSupportedPrimDescFactory({{firstCreatorType, precision}},
                         {{firstCreatorType, precision}},
                         attrs.implDescType);
    addSupportedPrimDescFactory({{secondCreatorType, precision}},
                         {{secondCreatorType, precision}},
                         attrs.implDescType);
    // canUseBlocked
    if (attrs.axis != 1) {
        addSupportedPrimDescFactory({{LayoutType::nCsp8c, precision}},
                             {{LayoutType::nCsp8c, precision}},
                             attrs.implDescType);
        addSupportedPrimDescFactory({{LayoutType::nCsp16c, precision}},
                             {{LayoutType::nCsp16c, precision}},
                             attrs.implDescType);
    }
}

void ShuffleChannels::createPrimitive() {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_SHCH_ERROR << "has not allocated destination memory";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_SHCH_ERROR << "has not allocated input memory";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_SHCH_ERROR << "has unidentified preferable primitive descriptor";

    const auto& memoryDesc = srcMemPtr->getDesc();
    attrs.spatialRank = attrs.dataRank - attrs.axis - 1;
    attrs.dataSize = memoryDesc.getPrecision().size();
    attrs.layoutType = memoryDesc.hasLayoutType(LayoutType::nCsp16c) ? LayoutType::nCsp16c :
                       memoryDesc.hasLayoutType(LayoutType::nCsp8c) ? LayoutType::nCsp8c :
                       memoryDesc.hasLayoutType(LayoutType::nspc) ? LayoutType::nspc : LayoutType::ncsp;

    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

void ShuffleChannels::prepareParams() {
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto builder = [this, &srcMemPtr](const ShuffleChannelsAttributes& key) -> std::shared_ptr<ShuffleChannelsExecutor> {
        dnnl::primitive_attr attr;
        auto selectedPD = getSelectedPrimitiveDescriptor();
        std::vector<MemoryDescPtr> srcDescs = {srcMemPtr->getDescPtr()};
        std::vector<MemoryDescPtr> dstDescs = {getChildEdgeAt(0)->getMemoryPtr()->getDescPtr()};
        auto shuffleChannelsExecutor = selectedPD->getExecutorFactoryAs<ShuffleChannelsExecutorFactory>()->makeExecutor(attrs, srcDescs, dstDescs, attr);
        selectedPD->setImplementationType(shuffleChannelsExecutor->getImplType());
        return shuffleChannelsExecutor;
    };
    attrs.srcDims = srcMemPtr->getStaticDims();
    attrs.srcBlockedDims = srcMemPtr->GetDescWithType<BlockedMemoryDesc>()->getBlockDims();

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(attrs, builder);
    if (!result.first) {
        IE_THROW() << "ShuffleChannelsExecutor was not found for node " << getName() << ".";
    }

    execPtr = result.first;
}

void ShuffleChannels::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void ShuffleChannels::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_SHCH_ERROR << "doesn't have a compiled executor.";

    int MB = (attrs.axis != 0) ? getParentEdgeAt(0)->getMemoryPtr()->getStaticDims()[0] : -1;

    execPtr->exec({getParentEdgeAt(0)->getMemoryPtr()}, {getChildEdgeAt(0)->getMemoryPtr()}, MB);
}

bool ShuffleChannels::created() const {
    return getType() == Type::ShuffleChannels;
}

void ShuffleChannels::addSupportedPrimDescFactory(const std::vector<PortConfigurator>& inPortConfigs,
                                                  const std::vector<PortConfigurator>& outPortConfigs,
                                                  impl_desc_type implType,
                                                  bool dynBatchSupport) {
    auto fill_port = [] (const PortConfigurator& portConfigurator, const Shape& shape,
                         InferenceEngine::Precision prc, std::vector<PortConfig>& port) -> bool {
        // In order to simplify particular node initialization logic we just don't add config in case target shape is not supported by blockedDescCreator.
        // This should be suitable for major of scenarios since almost all nodes add `ncsp` blockedDescCreator which supports any shape rank.
        if (shape.getRank() < portConfigurator.blockedDescCreator->getMinimalRank())
            return false;

        PortConfig portConfig;
        portConfig.inPlace(portConfigurator.inPlace);
        portConfig.constant(portConfigurator.constant);
        portConfig.setMemDesc(portConfigurator.blockedDescCreator->createSharedDesc(prc, shape));

        port.push_back(std::move(portConfig));

        return true;
    };

    NodeConfig config;
    for (size_t i = 0; i < inPortConfigs.size(); i++) {
        auto shape = inPortConfigs[i].shape.getRank() == 0 ? getInputShapeAtPort(i) : inPortConfigs[i].shape;
        auto prc = inPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalInputPrecisionAtPort(i) : inPortConfigs[i].prc;
        if (!fill_port(inPortConfigs[i], shape, prc, config.inConfs))
            return;
    }

    for (size_t i = 0; i < outPortConfigs.size(); i++) {
        auto dims = outPortConfigs[i].shape.getRank() == 0 ? getOutputShapeAtPort(i) : outPortConfigs[i].shape;
        auto prc = outPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalOutputPrecisionAtPort(i) : outPortConfigs[i].prc;
        if (!fill_port(outPortConfigs[i], dims, prc, config.outConfs))
            return;
    }

    std::vector<MemoryDescPtr> srcDescs = {config.inConfs[0].getMemDesc()};
    std::vector<MemoryDescPtr> dstDescs = {config.outConfs[0].getMemDesc()};
    auto factory = std::make_shared<ShuffleChannelsExecutorFactory>(attrs, srcDescs, dstDescs,
                                                                    std::make_shared<ExecutorContext>(context, getPrimitivesPriority()));
    supportedPrimitiveDescriptors.push_back({config, implType, factory});
}


}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
