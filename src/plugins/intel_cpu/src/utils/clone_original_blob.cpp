// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clone_original_blob.h"
#include <optional>

#include "cpu_memory.h"
#include "graph_context.h"
#include "cpu/x64/jit_generator.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/has_subnormals.h"
#include "openvino/core/parallel.hpp"
// #include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "openvino/core/type/element_type.hpp"
#include "ov_optional.hpp"
#include "dnnl_extension_utils.h"
#include "utils/debug_capabilities.h"

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

MemoryPtr cloneBlob(const IMemory& blob, const dnnl::engine& engine, bool needFlushDenormalsToZero) {
    const auto& memDesc = blob.getDesc();
    const auto prec = blob.getPrecision();
    const size_t size = blob.getShape().getElementsCount();
    MemoryPtr memory;

    // CVS-74980
    // oneDNN always allocate 1byte for element type with bitWidth < 8 (u4,u1...)
    // but ngraph Constant uses actual bitWidth for data storage allocation
    // in that case we make a copy to avoid overflow
    if (blob.getSize() >= memDesc.getCurrentMemSize()) {
        if (prec == element::string) {
            memory = std::make_shared<StringMemory>(engine, memDesc, blob.getDataAs<element::string>());
        } else {
            memory = std::make_shared<Memory>(engine, memDesc, blob.getData());
        }
    } else {
        if (prec == element::string) {
            memory = std::make_shared<StringMemory>(engine, memDesc);
            auto src = blob.getDataAs<StringMemory::OvString>();
            auto dst = memory->getDataAs<StringMemory::OvString>();
            std::copy(src, src + size, dst);
        } else {
            memory = std::make_shared<Memory>(engine, memDesc);
            memcpy(memory->getData(), blob.getData(), blob.getSize());
        }
    }

    MemoryPtr ptr;
    if (memDesc.getPrecision() == element::string) {
        ptr = std::make_shared<StringMemory>(engine, memDesc);
    } else {
        ptr = std::make_shared<StaticMemory>(engine, memDesc);
    }

    ptr->load(*memory.get(), needFlushDenormalsToZero);

    return ptr;
}

InputPrepType requiresPreProcessing(const IMemory& blob, GraphContext::CPtr context, const dnnl::engine& engine) {
    const auto shape = blob.getShape();
    const auto prec = blob.getPrecision();

    // DAZ has been set, processor automatically converts all denormal source operands
    // to a zero with the sign of the original operand before performing any
    // computations on them, thus no need to flush them to zero manually
    bool needFlushDenormalsToZero = context->getConfig().DAZOn ? false : true;

    auto isBlobAligned = [&] () {
        const void *ptr = blob.getData();
        bool blobAlignedOnSSE = true;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        // Majority of arithmetic and data processing instructions in legacy SSE isa requires
        // the memory address in the operands must be aligned on 16-byte boundary. To ensure
        // safely reusing ngraph const blob memory, need to check address alignment.
        blobAlignedOnSSE = mayiuse(dnnl::impl::cpu::x64::avx2) || ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0);
#endif
        const bool blobAlignedWithPrec = prec.size() > 1 ? (reinterpret_cast<size_t>(ptr) % prec.size()) == 0 : true;
        return blobAlignedWithPrec && blobAlignedOnSSE;
    };

    // @WARNING The order of the checks below matters
    // The checks are ordered from lightweight to heavy
    if (prec == element::string) {
        DEBUG_LOG("Clone is necessary for a string Constant");
        return InputPrepType::SimpleClone;
    }

    const bool mustFlushDenormalsToZero = needFlushDenormalsToZero && std::make_shared<HasSubnormals>()->execute(blob);
    if (mustFlushDenormalsToZero) {
        DEBUG_LOG("Clone is necessary for Constant containing subnormals");
        return InputPrepType::FTZ;
    }

    if (!isBlobAligned()) {
        DEBUG_LOG("Clone is necessary for not aligned blobs");
        return InputPrepType::SimpleClone;
    }

    if (context->getWeightsCache() &&
        context->getNumNumaNodes() > 1 &&
        context->getCPUStreamExecutor()->get_streams_num() > 1) {
        DEBUG_LOG("Clone is necessary for multistream multisocket configuration");
        return InputPrepType::PutToNumaLocalCache;
    }

    DEBUG_LOG("Clone is not required");

    return InputPrepType::None;
}

}   // namespace intel_cpu
}   // namespace ov
