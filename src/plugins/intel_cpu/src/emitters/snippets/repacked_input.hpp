// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_desc/cpu_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

struct RepackedInputKernel {
    RepackedInputKernel() = default;
    virtual ~RepackedInputKernel() = default;
    virtual void operator()(const void* args) const = 0;
};

struct RepackedInput {
    RepackedInput() = default;
    RepackedInput(std::shared_ptr<const RepackedInputKernel> kernel,
                  CpuBlockedMemoryDescPtr desc,
                  VectorDims in_offsets,
                  VectorDims out_offsets);

    template <class T = RepackedInputKernel,
              typename std::enable_if<std::is_base_of<RepackedInputKernel, T>::value, bool>::type = true>
    std::shared_ptr<const T> kernel() const {
        const auto ker = std::dynamic_pointer_cast<const T>(m_kernel);
        OPENVINO_ASSERT(ker, "Kernel is empty!");
        return ker;
    }

    const CpuBlockedMemoryDescPtr& desc() const;
    const VectorDims& in_offsets() const;
    const VectorDims& out_offsets() const;

private:
    std::shared_ptr<const RepackedInputKernel> m_kernel{nullptr};
    CpuBlockedMemoryDescPtr m_desc{nullptr};
    VectorDims m_in_offsets{};
    VectorDims m_out_offsets{};
};

}  // namespace intel_cpu
}  // namespace ov
