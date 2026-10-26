//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffer_metadata.hpp"
#include "math.hpp"
#include "npu_bytecode_utils/logger.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

std::optional<uint64_t> intel_npu::vm::computeMaxLinearOffset(const intel_npu::vm::BufferMetadata& meta) {
    if (meta.shape.size() != meta.strides.size()) {
        return std::nullopt;
    }
    uint64_t offset = 0;
    for (size_t i = 0; i < meta.shape.size(); ++i) {
        if (meta.shape.at(i) <= 0 || meta.strides.at(i) < 0) {
            return std::nullopt;
        }
        const auto extent = static_cast<uint64_t>(meta.shape.at(i) - 1);
        const auto stride = static_cast<uint64_t>(meta.strides.at(i));
        if (!checkedAddProduct(extent, stride, offset)) {
            return std::nullopt;
        }
    }
    return offset;
}

std::optional<uint64_t> intel_npu::vm::computeLogicalByteSize(const intel_npu::vm::BufferMetadata& meta) {
    const auto maxLinearOffset = computeMaxLinearOffset(meta);
    if (!maxLinearOffset.has_value() || meta.elemByteSize == 0) {
        return std::nullopt;
    }
    if (*maxLinearOffset == std::numeric_limits<uint64_t>::max()) {
        return std::nullopt;
    }
    const auto elementCount = *maxLinearOffset + 1;
    uint64_t byteSize = 0;
    if (!checkedMultiply(elementCount, static_cast<uint64_t>(meta.elemByteSize), byteSize)) {
        return std::nullopt;
    }
    return byteSize;
}

std::optional<uint64_t> intel_npu::vm::computeSubviewStartElement(const intel_npu::vm::BufferMetadata& srcMeta,
                                                                  const std::vector<int64_t>& offsets,
                                                                  const std::vector<int64_t>& sizes,
                                                                  const std::vector<int64_t>& strides) {
    const auto rank = offsets.size();
    if (sizes.size() != rank || strides.size() != rank) {
        NPU_VM_LOG_ERROR("buffer.subview expected offsets, sizes and strides to have the same rank, got {} for "
                         "offsets, {} for sizes and {} for strides",
                         rank, sizes.size(), strides.size());
        return std::nullopt;
    }
    if (srcMeta.shape.size() != rank || srcMeta.strides.size() != rank) {
        NPU_VM_LOG_ERROR("buffer.subview source metadata rank mismatch: subview rank={}, source shape rank={}, source "
                         "stride rank={}",
                         rank, srcMeta.shape.size(), srcMeta.strides.size());
        return std::nullopt;
    }

    uint64_t startElem = 0;
    for (size_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        if (offsets.at(dimIdx) < 0 || sizes.at(dimIdx) <= 0 || strides.at(dimIdx) < 0 ||
            srcMeta.strides.at(dimIdx) < 0 || offsets.at(dimIdx) >= srcMeta.shape.at(dimIdx)) {
            NPU_VM_LOG_ERROR("buffer.subview invalid offsets/sizes/strides at dim {}: offset={}, size={}, stride={}, "
                             "source shape={}, source stride={}",
                             dimIdx, offsets.at(dimIdx), sizes.at(dimIdx), strides.at(dimIdx), srcMeta.shape.at(dimIdx),
                             srcMeta.strides.at(dimIdx));
            return std::nullopt;
        }
        uint64_t stridedExtent = 0;
        if (!checkedMultiply(static_cast<uint64_t>(sizes.at(dimIdx) - 1), static_cast<uint64_t>(strides.at(dimIdx)),
                             stridedExtent) ||
            static_cast<uint64_t>(offsets.at(dimIdx)) > std::numeric_limits<uint64_t>::max() - stridedExtent) {
            NPU_VM_LOG_ERROR("buffer.subview index overflows at dim {}: offset={}, size={}, stride={}", dimIdx,
                             offsets.at(dimIdx), sizes.at(dimIdx), strides.at(dimIdx));
            return std::nullopt;
        }
        const auto lastIndex = static_cast<uint64_t>(offsets.at(dimIdx)) + stridedExtent;
        if (lastIndex >= static_cast<uint64_t>(srcMeta.shape.at(dimIdx))) {
            NPU_VM_LOG_ERROR("buffer.subview exceeds source shape at dim {}: last index={}, source shape={}", dimIdx,
                             lastIndex, srcMeta.shape.at(dimIdx));
            return std::nullopt;
        }
        const auto srcStride = static_cast<uint64_t>(srcMeta.strides.at(dimIdx));
        if (!checkedAddProduct(static_cast<uint64_t>(offsets.at(dimIdx)), srcStride, startElem)) {
            NPU_VM_LOG_ERROR("buffer.subview byte offset overflows at dim {}: offset={}, source stride={}", dimIdx,
                             offsets.at(dimIdx), srcStride);
            return std::nullopt;
        }
    }
    return startElem;
}

bool intel_npu::vm::validateShapeAndStrides(std::string_view opName, const intel_npu::vm::BufferMetadata& meta) {
    if (meta.shape.size() != meta.strides.size()) {
        NPU_VM_LOG_ERROR("{} metadata rank mismatch: shape rank={}, stride rank={}", opName, meta.shape.size(),
                         meta.strides.size());
        return false;
    }
    if (meta.shape.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        NPU_VM_LOG_ERROR("{} metadata rank exceeds int64_t range: rank={}", opName, meta.shape.size());
        return false;
    }
    const auto rank = static_cast<int64_t>(meta.shape.size());
    for (int64_t dim = 0; dim < rank; ++dim) {
        const auto dimIdx = static_cast<size_t>(dim);
        if (meta.shape.at(dimIdx) <= 0 || meta.strides.at(dimIdx) < 0) {
            NPU_VM_LOG_ERROR("{} requires positive shape and non-negative strides, got dim {}: shape={}, stride={}",
                             opName, dim, meta.shape.at(dimIdx), meta.strides.at(dimIdx));
            return false;
        }
    }
    return true;
}
