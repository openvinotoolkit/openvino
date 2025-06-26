// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "cpu_types.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

struct InputRepackerKernel {
    InputRepackerKernel() = default;
    virtual ~InputRepackerKernel() = default;
    virtual void operator()(const void* args) const = 0;
};

struct InputRepacker {
    InputRepacker() = default;
    InputRepacker(std::shared_ptr<const InputRepackerKernel> kernel,
                  CpuBlockedMemoryDescPtr desc,
                  VectorDims in_offsets,
                  VectorDims out_offsets);

    template <class T = InputRepackerKernel, std::enable_if_t<std::is_base_of_v<InputRepackerKernel, T>, bool> = true>
    [[nodiscard]] std::shared_ptr<const T> kernel() const {
        const auto ker = std::dynamic_pointer_cast<const T>(m_kernel);
        OPENVINO_ASSERT(ker, "Kernel is empty!");
        return ker;
    }

    [[nodiscard]] const CpuBlockedMemoryDescPtr& desc() const;
    [[nodiscard]] const VectorDims& in_offsets() const;
    [[nodiscard]] const VectorDims& out_offsets() const;

private:
    std::shared_ptr<const InputRepackerKernel> m_kernel{nullptr};
    CpuBlockedMemoryDescPtr m_desc{nullptr};
    VectorDims m_in_offsets;
    VectorDims m_out_offsets;
};

using InputRepackerMap = std::unordered_map<size_t, ov::intel_cpu::InputRepacker>;

}  // namespace ov::intel_cpu
