// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parameters.hpp"
#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "function.hpp"
#include "math.hpp"
#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/print_utils.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_runtime/npu_vm_runtime.hpp"

#include <climits>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace intel_npu::vm;

std::optional<BufferHandle> intel_npu::vm::extractMemRef(
        npu_vm_runtime_mem_ref_handle_t handle, BufferManager& bufferManager,
        std::unordered_map<BufferHandle, BufferMetadata>& bufferMetadata, Permission permission,
        const FuncParamResType& type, const std::vector<uint16_t>& typeByteSizes) {
    if (!std::holds_alternative<BufferType>(type.type.data)) {
        NPU_VM_LOG_ERROR("Unsupported parameter type for memref argument: expected BufferType, got type index {}",
                         type.typeSectionIndex);
        return std::nullopt;
    }
    auto bufferType = std::get<BufferType>(type.type.data);
    if (bufferType.dataTypeIndex >= static_cast<uint64_t>(typeByteSizes.size())) {
        NPU_VM_LOG_ERROR("Data type index {} out of bounds for type byte sizes table of size {}",
                         bufferType.dataTypeIndex, typeByteSizes.size());
        return std::nullopt;
    }
    const auto elemByteSize = typeByteSizes.at(bufferType.dataTypeIndex);

    if (handle == nullptr) {
        NPU_VM_LOG_ERROR("Null pointer provided for memref argument");
        return std::nullopt;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto& memRef = *reinterpret_cast<npu_vm_runtime_mem_ref*>(handle);
    if (memRef.basePtr == nullptr || memRef.data == nullptr) {
        NPU_VM_LOG_ERROR("Invalid memref argument: basePtr ({}) and data ({}) must not be null", memRef.basePtr,
                         memRef.data);
        return std::nullopt;
    }
    if (memRef.dimsCount == 0 || bufferType.rank != memRef.dimsCount) {
        NPU_VM_LOG_ERROR(
                "Invalid memref argument: dimsCount is 0 or does not match bufferType rank (dimsCount: {}, rank: {})",
                memRef.dimsCount, bufferType.rank);
        return std::nullopt;
    }
    if (memRef.dimsCount > static_cast<int64_t>(memRef.sizes.size()) ||
        memRef.dimsCount > static_cast<int64_t>(memRef.strides.size())) {
        NPU_VM_LOG_ERROR("Invalid memref argument: dimsCount ({}) exceeds sizes/strides vector capacity (sizes: {}, "
                         "strides: {})",
                         memRef.dimsCount, memRef.sizes.size(), memRef.strides.size());
        return std::nullopt;
    }
    int64_t bufferByteSize = 0;
    if (!checkedMultiplyNonNegative(memRef.sizes.at(0), memRef.strides.at(0), bufferByteSize) ||
        !checkedMultiplyNonNegative(bufferByteSize, elemByteSize, bufferByteSize)) {
        NPU_VM_LOG_ERROR(
                "Buffer byte size calculation overflow for memref argument with sizes {}, strides {}, elemByteSize {}",
                formatVector(memRef.sizes), formatVector(memRef.strides), elemByteSize);
        return std::nullopt;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* address = static_cast<uint8_t*>(const_cast<void*>(memRef.data));
    auto bufferHandle = bufferManager.createFromMemory(address, static_cast<size_t>(bufferByteSize), permission);

    BufferMetadata metadata{static_cast<int64_t>(bufferType.dataTypeIndex), memRef.sizes, memRef.strides, elemByteSize};
    bufferMetadata.emplace(bufferHandle, metadata);

    return bufferHandle;
}
