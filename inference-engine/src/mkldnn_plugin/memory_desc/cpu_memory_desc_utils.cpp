// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_memory.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include <limits>
#include <vector>
#include <numeric>
#include <blob_factory.hpp>
#include <dnnl_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

DnnlMemoryDescPtr MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDescPtr &desc) {
    if (MemoryDescType::Blocked == desc->getType()) {
        const auto cpuDesc = desc->as<CpuBlockedMemoryDesc>();
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(),
                                                        cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                                        cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides()));
    } else if (MemoryDescType::Mkldnn & desc->getType()) {
        return std::dynamic_pointer_cast<DnnlMemoryDesc>(desc);
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to DnnlMemoryDesc";
    }
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::DnnlBlocked == desc.getType()) {
        return DnnlBlockedMemoryDesc(*desc.as<DnnlBlockedMemoryDesc>());
    } else if (MemoryDescType::Blocked == desc.getType()) {
        const auto cpuDesc = desc.as<CpuBlockedMemoryDesc>();
        return DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(), cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                     cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides());
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to DnnlMemoryDesc";
    }
}

CpuBlockedMemoryDesc MemoryDescUtils::convertToCpuBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to CpuBlockedMemoryDesc";
    const auto &blkDesc = desc.getBlockingDesc();
    return CpuBlockedMemoryDesc(desc.getPrecision(), Shape(desc.getDims()), blkDesc.getBlockDims(), blkDesc.getOrder(), blkDesc.getOffsetPadding(),
                                blkDesc.getOffsetPaddingToData(), blkDesc.getStrides());
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    const auto &blkDesc = desc.getBlockingDesc();
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to DnnlBlockedMemoryDesc";
    return DnnlBlockedMemoryDesc(desc.getPrecision(), Shape(desc.getDims()), blkDesc.getBlockDims(), blkDesc.getOrder(), blkDesc.getOffsetPadding(),
                                 blkDesc.getOffsetPaddingToData(), blkDesc.getStrides());
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr &desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
    }
}

InferenceEngine::Blob::Ptr MemoryDescUtils::interpretAsBlob(const MKLDNNMemory &mem) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    return make_blob_with_precision(desc, mem.GetData());
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return InferenceEngine::TensorDesc(blockingDesc->getPrecision(), blockingDesc->getShape().getStaticDims(),
                                           {blockingDesc->getBlockDims(), blockingDesc->getOrder(), blockingDesc->getOffsetPadding(),
                                            blockingDesc->getOffsetPaddingToData(), blockingDesc->getStrides()});
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to InferenceEngine::TensorDesc";
    }
}

std::string MemoryDescUtils::dim2str(Dim dim) {
    return dim == Shape::UNDEFINED_DIM ? "?" : std::to_string(dim);
}

std::string MemoryDescUtils::dims2str(const VectorDims& dims) {
    std::stringstream output;
    output << "{";

    if (!dims.empty()) {
        auto itr = dims.begin();
        do {
            output << dim2str(*itr);
        } while (++itr != dims.end() && output << ", ");
    }

    output << "}";
    return output.str();
}

std::shared_ptr<MemoryDesc> MemoryDescUtils::makeDummyDesc(const MemoryDesc &desc, Dim dummyVal) {
    auto dummyShape = makeDummyShape(desc.getShape(), dummyVal);
    return desc.cloneWithNewDims(dummyShape.getStaticDims());
}

Shape MemoryDescUtils::makeDummyShape(const Shape &shape, Dim dummyVal) {
    // relaxed from getMaxDims() to getDims(), because maybe overflow after computation on upper bound value
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    std::transform(dims.begin(), dims.end(), dummyDims.begin(), [=](const Dim& dim){ return dim == Shape::UNDEFINED_DIM ? dummyVal : dim; });
    return Shape(dummyDims);
}

} // namespace MKLDNNPlugin
