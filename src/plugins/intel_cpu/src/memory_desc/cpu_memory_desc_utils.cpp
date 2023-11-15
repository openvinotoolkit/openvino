// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "ie_ngraph_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <cpu_memory.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include <limits>
#include <vector>
#include <numeric>
#include <blob_factory.hpp>
#include <dnnl_types.h>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

DnnlMemoryDescPtr MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDescPtr &desc) {
    if (MemoryDescType::Blocked == desc->getType()) {
        const auto cpuDesc = desc->as<CpuBlockedMemoryDesc>();
        return std::shared_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(),
                                                        cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                                        cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides()));
    } else if (MemoryDescType::Dnnl & desc->getType()) {
        return std::dynamic_pointer_cast<DnnlMemoryDesc>(desc);
    } else {
        OPENVINO_THROW("Cannot convert MemoryDesc to DnnlMemoryDesc");
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
        OPENVINO_THROW("Cannot convert MemoryDesc to DnnlMemoryDesc");
    }
}

CpuBlockedMemoryDesc MemoryDescUtils::convertToCpuBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to CpuBlockedMemoryDesc";

    const auto& blkDesc = desc.getBlockingDesc();
    const auto& dims = desc.getDims();

    auto strides = blkDesc.getStrides();
    // for empty tensor case InferenceEngine::TensorDesc fill strides with non zero values before first 0 dims
    // i.e. dims[1, 0, 2, 3] -> strides [0, 6, 3, 1]
    if (std::any_of(dims.begin(), dims.end(), [](size_t dim){ return dim == 0; })) {
        std::fill(strides.begin(), strides.end(), 0);
    }

    return CpuBlockedMemoryDesc(InferenceEngine::details::convertPrecision(desc.getPrecision()),
                                Shape(dims),
                                blkDesc.getBlockDims(),
                                blkDesc.getOrder(),
                                blkDesc.getOffsetPadding(),
                                blkDesc.getOffsetPaddingToData(),
                                strides);
}

DnnlBlockedMemoryDesc MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        IE_THROW() << "Cannot convert InferenceEngine::TensorDesc with ANY layout to DnnlBlockedMemoryDesc";

    const auto& blkDesc = desc.getBlockingDesc();
    const auto& dims = desc.getDims();

    auto strides = blkDesc.getStrides();
    // for empty tensor case InferenceEngine::TensorDesc fill strides with non zero values before first 0 dims
    // i.e. dims[1, 0, 2, 3] -> strides [0, 6, 3, 1]
    if (std::any_of(dims.begin(), dims.end(), [](size_t dim){ return dim == 0; })) {
        std::fill(strides.begin(), strides.end(), 0);
    }

    return DnnlBlockedMemoryDesc(InferenceEngine::details::convertPrecision(desc.getPrecision()),
                                 Shape(desc.getDims()),
                                 blkDesc.getBlockDims(),
                                 blkDesc.getOrder(),
                                 blkDesc.getOffsetPadding(),
                                 blkDesc.getOffsetPaddingToData(),
                                 strides);
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr &desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    } else {
        OPENVINO_THROW("Can not convert unsupported memory descriptor");
    }
}

InferenceEngine::Blob::Ptr MemoryDescUtils::interpretAsBlob(const IMemory &mem) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    return make_blob_with_precision(desc, mem.getData());
}

InferenceEngine::TensorDesc MemoryDescUtils::interpretAsBlobDesc(const IMemory &mem) {
    auto& memDesc = mem.getDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    return InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        InferenceEngine::BlockingDesc blkDesc =
            desc.getShape().hasZeroDims() ? InferenceEngine::BlockingDesc(blockingDesc->getBlockDims(),
                                                                          blockingDesc->getOrder(),
                                                                          blockingDesc->getOffsetPadding(),
                                                                          blockingDesc->getOffsetPaddingToData())
                                          : InferenceEngine::BlockingDesc(blockingDesc->getBlockDims(),
                                                                          blockingDesc->getOrder(),
                                                                          blockingDesc->getOffsetPadding(),
                                                                          blockingDesc->getOffsetPaddingToData(),
                                                                          blockingDesc->getStrides());
        return InferenceEngine::TensorDesc(InferenceEngine::details::convertPrecision(blockingDesc->getPrecision()),
                                           blockingDesc->getShape().getStaticDims(),
                                           blkDesc);
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
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] = dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVal)) : dims[i];
    }
    return Shape(dummyDims);
}

Shape MemoryDescUtils::makeDummyShape(const Shape &shape, const VectorDims& dummyVals) {
    if (shape.getRank() != dummyVals.size()) {
        OPENVINO_THROW("makeDummyShape(): dummyVals vector size and shape ranks mismatch");
    }
    const auto& minDims = shape.getMinDims();
    const auto& maxDims = shape.getMaxDims();
    const auto& dims = shape.getDims();
    VectorDims dummyDims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        dummyDims[i] = dims[i] == Shape::UNDEFINED_DIM ? std::min(maxDims[i], std::max(minDims[i], dummyVals[i])) : dims[i];
    }
    return Shape(dummyDims);
}
}   // namespace intel_cpu
}   // namespace ov
