// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_desc/cpu_blocked_memory_desc.h"

namespace ov::intel_cpu {

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

    template <class T = RepackedInputKernel, std::enable_if_t<std::is_base_of_v<RepackedInputKernel, T>, bool> = true>
    std::shared_ptr<const T> kernel() const {
        const auto ker = std::dynamic_pointer_cast<const T>(m_kernel);
        OPENVINO_ASSERT(ker, "Kernel is empty!");
        return ker;
    }

    [[nodiscard]] const CpuBlockedMemoryDescPtr& desc() const;
    [[nodiscard]] const VectorDims& in_offsets() const;
    [[nodiscard]] const VectorDims& out_offsets() const;

private:
    std::shared_ptr<const RepackedInputKernel> m_kernel{nullptr};
    CpuBlockedMemoryDescPtr m_desc{nullptr};
    VectorDims m_in_offsets{};
    VectorDims m_out_offsets{};
};

}  // namespace ov::intel_cpu
