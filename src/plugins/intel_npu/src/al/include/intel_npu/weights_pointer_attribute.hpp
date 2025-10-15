// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/runtime_attribute.hpp"

namespace intel_npu {

/**
 * @brief Attribute containing the memory address of a weights buffer and the size of the buffer in bytes.
 * @details Used as part of the serialization/deserialization algorithms in order to allow processing models without
 * copying weights.
 */
class WeightsPointerAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightsPointerAttribute", "0", RuntimeAttribute);

    WeightsPointerAttribute() = delete;

    WeightsPointerAttribute(const void* pointer, const size_t size)
        : memory_pointer(reinterpret_cast<size_t>(pointer)),
          byte_size(size) {}

    /**
     * @note The names of the attributes have been kept short in order to save some memory (there may be a lot of
     * "ov::Constant" nodes in a model). Also, two characters should be sufficient to avoid collisions. "mp" stands for
     * "memory pointer", "ms" for "memory size".
     */
    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mp", memory_pointer);
        visitor.on_attribute("ms", byte_size);
        return true;
    }

    size_t memory_pointer;
    size_t byte_size;
};

}  // namespace intel_npu
