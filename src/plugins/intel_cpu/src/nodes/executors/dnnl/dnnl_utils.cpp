// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_utils.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <unordered_map>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/reorder.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "weights_cache.hpp"

namespace ov::intel_cpu::utils {

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const ExecutorContext::CPtr& context,
                               const bool needShiftSignedToUnsigned) {
    const auto privateWeightCache = context->getPrivateWeightCache();
    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");

    return prepareWeightsMemory(srcWeightDesc,
                                dstWeightDesc,
                                weightsMem,
                                context->getEngine(),
                                context->getRuntimeCache(),
                                context->getWeightsCache(),
                                privateWeightCache,
                                needShiftSignedToUnsigned);
}

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr& srcWeightDesc,
                               const DnnlMemoryDescPtr& dstWeightDesc,
                               const MemoryCPtr& weightsMem,
                               const dnnl::engine& eng,
                               const MultiCachePtr& rtCache,
                               const WeightsSharing::Ptr& globalWeightCache,
                               const std::shared_ptr<std::unordered_map<std::string, MemoryPtr>>& privateWeightCache,
                               bool needShiftSignedToUnsigned) {
    const auto format = dstWeightDesc->serializeFormat();
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
                OPENVINO_THROW("Unsupported data type for shiftting sign to unsign");
            }
            return _ptr;
        }

        Memory srcMemory{eng, srcWeightDesc, weightsMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
        node::Reorder::reorderData(srcMemory, *_ptr, rtCache);

        return _ptr;
    };

    MemoryPtr ptr;
    if (globalWeightCache && dnnl::memory::format_kind::blocked == dstWeightDesc->getDnnlDesc().get_format_kind()) {
        ptr = MemoryPtr(
            *globalWeightCache->findOrCreate(DnnlExtensionUtils::computeWeightsStringHash(weightsMem, dstWeightDesc),
                                             create));
    } else {
        ptr = create();
    }

    if (privateWeightCache) {
        (*privateWeightCache)[format] = ptr;
    }

    return ptr;
}

}  // namespace ov::intel_cpu::utils
