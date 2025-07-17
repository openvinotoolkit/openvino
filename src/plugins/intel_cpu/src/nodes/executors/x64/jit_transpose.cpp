// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_transpose.hpp"

#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/permute_kernel.h"
#include "nodes/executors/transpose.hpp"
#include "openvino/core/except.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {
void JitTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    OPENVINO_ASSERT(pKernel, "Could not execute. Kernel for Transpose node was not compiled.");
    const auto* srcData = src[0]->getDataAs<const uint8_t>();
    auto* dstData = dst[0]->getDataAs<uint8_t>();
    const int MB = src[0]->getStaticDims()[0];

    pKernel->execute(srcData, dstData, MB);
}

bool JitTransposeExecutor::init(const TransposeParams& transposeParams,
                                [[maybe_unused]] const std::vector<MemoryDescPtr>& srcDescs,
                                [[maybe_unused]] const std::vector<MemoryDescPtr>& dstDescs,
                                [[maybe_unused]] const dnnl::primitive_attr& attr) {
    pKernel = std::make_shared<PermuteKernel>(transposeParams.permuteParams);
    return true;
}

bool JitTransposeExecutorBuilder::isSupported([[maybe_unused]] const TransposeParams& transposeParams,
                                              [[maybe_unused]] const std::vector<MemoryDescPtr>& srcDescs,
                                              [[maybe_unused]] const std::vector<MemoryDescPtr>& dstDescs) const {
#if defined(OPENVINO_ARCH_X86_64)
    return mayiuse(x64::sse41);
#else
    return false;
#endif  // OPENVINO_ARCH_X86_64
}

}  // namespace ov::intel_cpu
