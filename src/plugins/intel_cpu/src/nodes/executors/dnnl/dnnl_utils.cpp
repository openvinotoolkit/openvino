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

DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc, const DnnlMemoryDescPtr dstDesc) {
    const auto& weiDesc = srcDesc->getDnnlDesc();
    const auto reorderedWeiDesc = dnnl::memory::desc{weiDesc.get_dims(), weiDesc.get_data_type(), dnnl::memory::format_tag::ba};
    const auto transposedWeiDesc = reorderedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                               const DnnlMemoryDescPtr dstWeightDesc,
                               const MemoryCPtr weightsMem,
                               const ExecutorContext::CPtr context) {
    const auto& eng = context->getEngine();
    const auto& format = dstWeightDesc->serializeFormat();

    const auto privateWeightCache = context->getPrivateWeighCache();
    if (privateWeightCache) {
        auto itr = privateWeightCache->find(format);
        if (privateWeightCache->end() != itr) {
            return itr->second;
        }
    }

    auto create = [&]() {
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
                                        std::to_string(*weightsMem->getDataAs<uint64_t>());
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
