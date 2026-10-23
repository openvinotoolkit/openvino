// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "function.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace ov {

namespace intel_npu::vm {

struct MemRef {
    enum class DirtyFlag : uint32_t {
        NONE = 0,
        BUFFER_PTR = 0x01,
        OFFSET = 0x02,
        SIZES = 0x04,
        STRIDES = 0x08,
        ALL = 0xFF
    };

    const void* basePtr = nullptr;
    const void* data = nullptr;
    int64_t offset = 0;
    int64_t dimsCount = 0;
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    uint32_t dirtyFlags = static_cast<uint32_t>(DirtyFlag::NONE);

    explicit MemRef(int64_t dims): dimsCount(dims), sizes(dims, 0), strides(dims, 0) {
    }

    void resetDirtyBits() {
        dirtyFlags = static_cast<uint32_t>(DirtyFlag::NONE);
    }
};

std::optional<BufferHandle> extractMemRef(npu_vm_runtime_mem_ref_handle_t handle, BufferManager& bufferManager,
                                          std::unordered_map<BufferHandle, BufferMetadata>& metadata,
                                          Permission permission, const FuncParamResType& type,
                                          const std::vector<uint16_t>& typeByteSizes);

}  // namespace intel_npu::vm

}  // namespace ov

using npu_vm_runtime_mem_ref = ov::intel_npu::vm::MemRef;
