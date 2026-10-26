//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vm_export.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace intel_npu::vm {

// Bytecode-level metadata for a buffer handle. The underlying memory is owned by the BufferManager;
// this struct tracks the rank, shape, strides and element-type index that the bytecode operates on.
struct NPU_VM_EXPORT BufferMetadata {
    int64_t elemTypeIndex{};       // Index into the type section
    std::vector<int64_t> shape;    // Shape per dimension (rank == shape.size())
    std::vector<int64_t> strides;  // Stride (in elements) per dimension; same length as shape
    uint16_t elemByteSize{};       // Element size in bytes, derived from elemTypeIndex on creation
};

std::optional<uint64_t> NPU_VM_EXPORT computeMaxLinearOffset(const BufferMetadata& meta);
std::optional<uint64_t> NPU_VM_EXPORT computeLogicalByteSize(const BufferMetadata& meta);
std::optional<uint64_t> NPU_VM_EXPORT computeSubviewStartElement(const BufferMetadata& srcMeta,
                                                                 const std::vector<int64_t>& offsets,
                                                                 const std::vector<int64_t>& sizes,
                                                                 const std::vector<int64_t>& strides);
bool NPU_VM_EXPORT validateShapeAndStrides(std::string_view opName, const BufferMetadata& meta);

}  // namespace intel_npu::vm
