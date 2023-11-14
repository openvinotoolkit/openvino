// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/cpu_memory_desc_utils.h"

#include "cpu_memory_desc.h"
#include "ie_ngraph_utils.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

#include <cpu_memory.h>
#include <dnnl_types.h>
#include <limits>
#include <numeric>
#include <vector>

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

CpuBlockedMemoryDesc MemoryDescUtils::createCpuBlockedMemoryDesc(const ov::SoPtr<ITensor>& tensor,
                                                                 const bool canEmptyShape) {
    auto element_type = tensor->get_element_type();
    auto shape = tensor->get_shape();
    if (shape.empty() && !canEmptyShape)
        shape = {tensor->get_size()};
    std::vector<size_t> blk_order(shape.size());
    std::iota(blk_order.begin(), blk_order.end(), 0);
    std::vector<size_t> dim_offset(shape.size(), 0);
    std::vector<size_t> blk_strides;
    auto byte_strides = element_type.bitwidth() >= 8 ? tensor->get_strides() : Strides{};
    if (byte_strides.empty()) {
        blk_strides = ov::row_major_strides(shape);
    } else if (tensor->get_size() == 0) {
        blk_strides.resize(shape.size());
    } else {
        // ROI tensor need figure out correct blk_strides
        blk_strides.resize(byte_strides.size());
        std::transform(byte_strides.begin(),
                       byte_strides.end(),
                       blk_strides.begin(),
                       [&element_type](size_t byte_stride) {
                           OPENVINO_ASSERT(byte_stride % element_type.size() == 0,
                                           "Limitation: Stride in bytes ",
                                           byte_stride,
                                           " should be divisible by size of element ",
                                           element_type.size());
                           return byte_stride / element_type.size();
                       });
    }
    return CpuBlockedMemoryDesc(InferenceEngine::details::convertPrecision(element_type),
                                Shape(shape),
                                shape,
                                blk_order,
                                0,
                                dim_offset,
                                blk_strides);
}

BlockedMemoryDescPtr MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDescPtr &desc) {
    if (desc->getType() & MemoryDescType::Blocked) {
        return std::dynamic_pointer_cast<BlockedMemoryDesc>(desc);
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
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
        IE_THROW() << "makeDummyShape(): dummyVals vector size and shape ranks mismatch";
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
