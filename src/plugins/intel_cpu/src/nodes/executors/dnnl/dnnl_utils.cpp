// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/reorder.h"

namespace ov {
namespace intel_cpu {
namespace utils {

template <>
DnnlMemoryDescPtr makeTransposedWeightDescriptor<DnnlFCPrimitive>(const DnnlMemoryDescPtr srcDesc,
                                                                  const DnnlMemoryDescPtr dstDesc,
                                                                  bool weightsNonTransposed) {
    if (!weightsNonTransposed)
        return srcDesc;

    const auto& weiDesc = srcDesc->getDnnlDesc();
    const auto reorderedWeiDesc = dnnl::memory::desc{weiDesc.get_dims(), weiDesc.get_data_type(), dnnl::memory::format_tag::ba};
    const auto transposedWeiDesc = reorderedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

template <>
DnnlMemoryDescPtr makeTransposedWeightDescriptor<DnnlMatMulPrimitive>(const DnnlMemoryDescPtr srcDesc,
                                                                      const DnnlMemoryDescPtr dstDesc,
                                                                      bool weightsNonTransposed) {
    const auto& weiDesc = srcDesc->getDnnlDesc();
    auto wDims = weiDesc.get_dims();
    auto wDataType = weiDesc.get_data_type();
    std::swap(wDims[wDims.size() - 1], wDims[wDims.size() - 2]);

    const auto format = weightsNonTransposed ? dnnl::memory::format_tag::ab : dnnl::memory::format_tag::ba;
    const auto transposedWeiDesc = dnnl::memory::desc{wDims, wDataType, format};

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                               const DnnlMemoryDescPtr dstWeightDesc,
                               const MemoryCPtr weightsMem,
                               const ExecutorContext::CPtr context,
                               const bool needShiftSignedToUnsigned) {
    const auto& eng = context->getEngine();
    const auto& format = dstWeightDesc->serializeFormat();

    const auto privateWeightCache = context->getPrivateWeighCache();
    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");
    if (privateWeightCache) {
        auto itr = privateWeightCache->find(format);
        if (privateWeightCache->end() != itr) {
            return itr->second;
        }
    }

    auto create = [&]() {
        // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html?highlight=128#inputs-of-the-same-type-s8
        auto src_wdt = srcWeightDesc->getPrecision();
        auto dst_wdt = dstWeightDesc->getPrecision();
        if (needShiftSignedToUnsigned && src_wdt.is_integral_number() && src_wdt.is_signed() &&
            dst_wdt.is_integral_number() && !dst_wdt.is_signed()) {
            assert(src_wdt.bitwidth() == dst_wdt.bitwidth());

            // prevent reorderData from doing conversion
            Memory srcMemory{eng, srcWeightDesc->cloneWithNewPrecision(dst_wdt), weightsMem->getData()};
            MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
            auto rtCache = context->getRuntimeCache();
            node::Reorder::reorderData(srcMemory, *_ptr, rtCache);

            // do shift
            auto count = _ptr->getSize() / _ptr->getDesc().getPrecision().size();
            if (dst_wdt == ov::element::u8) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    data[i] = data[i] + 128;
                }
            } else if (dst_wdt == ov::element::u4) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    auto low = (data[i] & 0xF) + 8;
                    auto high = (data[i] >> 4) + 8;
                    data[i] = (high << 4) | (low & 0xF);
                }
            } else {
                OPENVINO_ASSERT(false, "Unsupported data type for shiftting sign to unsign");
            }
            return _ptr;
        }

        Memory srcMemory{eng, srcWeightDesc, weightsMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
        auto rtCache = context->getRuntimeCache();
        node::Reorder::reorderData(srcMemory, *_ptr, rtCache);

        return _ptr;
    };

    auto globalWeightCache = context->getWeightsCache();
    MemoryPtr ptr;
    if (globalWeightCache &&
        dnnl::memory::format_kind::blocked == dstWeightDesc->getDnnlDesc().get_format_kind()) {
        const std::string string_hash = format + "_" + std::to_string(weightsMem->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(weightsMem->getData()));
        ptr = *globalWeightCache->findOrCreate(string_hash, create);
    } else {
        ptr = create();
    }

    (*privateWeightCache)[format] = ptr;

    return ptr;
}

}  // namespace utils
}  // namespace intel_cpu
}  // namespace ov
