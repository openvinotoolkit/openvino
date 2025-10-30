//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>

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
     * "ov::Constant" nodes in a model). Also, three characters should be sufficient to avoid collisions.
     */
    static constexpr const std::string_view POINTER_KEY = "mpZ";
    static constexpr const std::string_view BYTE_SIZE_KEY = "msZ";

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute(POINTER_KEY.data(), memory_pointer);
        visitor.on_attribute(POINTER_KEY.data(), byte_size);
        return true;
    }

    size_t memory_pointer;
    size_t byte_size;
};

}  // namespace intel_npu
