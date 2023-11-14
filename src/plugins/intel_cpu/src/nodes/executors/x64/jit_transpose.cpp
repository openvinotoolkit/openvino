// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_transpose.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
void JitTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) {
    if (!pKernel)
        OPENVINO_THROW("Could not execute. Kernel for Transpose node was not compiled.");

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(src[0]->getData());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dst[0]->getData());

    pKernel->execute(srcData, dstData, MB);
}

bool JitTransposeExecutor::init(const TransposeParams &transposeParams,
                                const std::vector<MemoryDescPtr> &srcDescs,
                                const std::vector<MemoryDescPtr> &dstDescs, const dnnl::primitive_attr &attr) {
    pKernel = std::make_shared<PermuteKernel>(transposeParams.permuteParams);
    return true;
}

bool JitTransposeExecutorBuilder::isSupported(const TransposeParams& transposeParams,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs) const {
#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(x64::sse41)) {
        return true;
    }
#endif // OPENVINO_ARCH_X86_64
    return false;
}

}   // namespace intel_cpu
}   // namespace ov