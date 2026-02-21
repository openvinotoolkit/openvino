// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_transpose.hpp"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "nodes/common/permute_kernel.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "openvino/core/except.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
JitTransposeExecutor::JitTransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context)
    : TransposeExecutor(attrs, std::move(context)) {}

void JitTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    OPENVINO_ASSERT(pKernel, "Could not execute. Kernel for Transpose node was not compiled.");
    const auto* srcData = src[0]->getDataAs<const uint8_t>();
    auto* dstData = dst[0]->getDataAs<uint8_t>();
    const int MB = src[0]->getStaticDims()[0];

    pKernel->execute(srcData, dstData, MB, context->getCpuParallel());
}

bool JitTransposeExecutor::init([[maybe_unused]] const MemoryArgs& memory) {
    pKernel = std::make_shared<PermuteKernel>(permuteParams);
    return true;
}

bool JitTransposeExecutor::supports([[maybe_unused]] const TransposeConfig& config) {
#if defined(OPENVINO_ARCH_X86_64)
    return mayiuse(x64::sse41);
#else
    return false;
#endif  // OPENVINO_ARCH_X86_64
}

ExecutorPtr JitTransposeExecutor::create(const TransposeAttrs& attrs,
                                         [[maybe_unused]] const MemoryArgs& memory,
                                         const ExecutorContext::CPtr& context) {
    return std::make_shared<JitTransposeExecutor>(attrs, context);
}

}  // namespace ov::intel_cpu
