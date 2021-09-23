// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/cpu_memory_desc_utils.h"

#include <dnnl_types.h>

#include <blob_factory.hpp>
#include <limits>
#include <numeric>
#include <vector>

#include "cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "mkldnn_memory.h"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

DnnlMemoryDescPtr MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDescPtr& desc) {
    if (MemoryDescType::Blocked == desc->getType()) {
        const auto cpuDesc = desc->as<CpuBlockedMemoryDesc>();
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(),
                                                                                cpuDesc->getShape(),
                                                                                cpuDesc->getBlockDims(),
                                                                                cpuDesc->getOrder(),
                                                                                cpuDesc->getOffsetPadding(),
                                                                                cpuDesc->getOffsetPaddingToData(),
                                                                                cpuDesc->getStrides()));
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
        return DnnlBlockedMemoryDesc(cpuDesc->getPrecision(),
                                     cpuDesc->getShape(),
                                     cpuDesc->getBlockDims(),
                                     cpuDesc->getOrder(),
                                     cpuDesc->getOffsetPadding(),
                                     cpuDesc->getOffsetPaddingToData(),
                                     cpuDesc->getStrides());
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to DnnlMemoryDesc";
    }
}

CpuBlockedMemoryDesc MemoryDescUtils::convertToCpuBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to CpuBlockedMemoryDesc";
    const auto& blkDesc = desc.getBlockingDesc();
    return CpuBlockedMemoryDesc(desc.getPrecision(),
                                Shape(desc.getDims()),
                                blkDesc.getBlockDims(),
                                blkDesc.getOrder(),
                                blkDesc.getOffsetPadding(),
                                blkDesc.getOffsetPaddingToData(),
                                blkDesc.getStrides());
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    const auto& blkDesc = desc.getBlockingDesc();
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to DnnlBlockedMemoryDesc";
    return DnnlBlockedMemoryDesc(desc.getPrecision(),
                                 Shape(desc.getDims()),
                                 blkDesc.getBlockDims(),
                                 blkDesc.getOrder(),
                                 blkDesc.getOffsetPadding(),
                                 blkDesc.getOffsetPaddingToData(),
                                 blkDesc.getStrides());
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr& desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
    }
}

MemoryDescPtr MemoryDescUtils::cloneWithUndefStridesAndOffset(const MemoryDesc& desc) {
    if (desc.getType() == MemoryDescType::Mkldnn) {
        IE_THROW() << "Can't apply undefined offset for mkldnn memory desc";
    }

    const auto blkMemDesc = desc.as<BlockedMemoryDesc>();

    VectorDims strides;
    VectorDims offsetPaddingToData;
    strides.resize(blkMemDesc->getBlockDims().size(), Shape::UNDEFINED_DIM);
    offsetPaddingToData.resize(blkMemDesc->getBlockDims().size(), 0);
    size_t offsetPadding = Shape::UNDEFINED_DIM;

    if (blkMemDesc->getType() == MemoryDescType::Blocked) {
        return std::make_shared<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(),
                                                      blkMemDesc->getShape(),
                                                      blkMemDesc->getBlockDims(),
                                                      blkMemDesc->getOrder(),
                                                      offsetPadding,
                                                      offsetPaddingToData,
                                                      strides);
    } else if (blkMemDesc->getType() == MemoryDescType::DnnlBlocked) {
        return DnnlBlockedMemoryDescPtr(new DnnlBlockedMemoryDesc(blkMemDesc->getPrecision(),
                                                                  blkMemDesc->getShape(),
                                                                  blkMemDesc->getBlockDims(),
                                                                  blkMemDesc->getOrder(),
                                                                  offsetPadding,
                                                                  offsetPaddingToData,
                                                                  strides));
    } else {
        IE_THROW() << "Cannot apply undefined offset. Unsupported memory desc type";
    }
}

MemoryDescPtr MemoryDescUtils::cloneWithDefaultStridesAndOffset(const MemoryDesc& desc) {
    const auto blkMemDesc = desc.as<BlockedMemoryDesc>();

    if (MemoryDescType::Blocked == desc.getType()) {
        return std::make_shared<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(),
                                                      blkMemDesc->getShape(),
                                                      blkMemDesc->getBlockDims(),
                                                      blkMemDesc->getOrder());
    } else if (MemoryDescType::DnnlBlocked == desc.getType()) {
        return DnnlBlockedMemoryDescPtr(new DnnlBlockedMemoryDesc(blkMemDesc->getPrecision(),
                                                                  blkMemDesc->getShape(),
                                                                  blkMemDesc->getBlockDims(),
                                                                  blkMemDesc->getOrder()));
    } else {
        IE_THROW() << "cloneWithDefaultStridesAndOffset supports Blocked descriptors only";
    }
}

MemoryDescPtr MemoryDescUtils::cloneWithNewPrecision(const MemoryDesc& desc, const InferenceEngine::Precision prec) {
    MemoryDescPtr newDesc = desc.clone();
    newDesc->setPrecision(prec);
    return newDesc;
}

InferenceEngine::Blob::Ptr MemoryDescUtils::interpretAsBlob(const MKLDNNMemory& mem) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    return make_blob_with_precision(desc, mem.GetData());
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return InferenceEngine::TensorDesc(blockingDesc->getPrecision(),
                                           blockingDesc->getShape().getStaticDims(),
                                           {blockingDesc->getBlockDims(),
                                            blockingDesc->getOrder(),
                                            blockingDesc->getOffsetPadding(),
                                            blockingDesc->getOffsetPaddingToData(),
                                            blockingDesc->getStrides()});
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

    auto itr = dims.begin();
    do {
        output << dim2str(*itr);
    } while (++itr != dims.end() && output << ", ");

    output << "}";
    return output.str();
}

}  // namespace MKLDNNPlugin
