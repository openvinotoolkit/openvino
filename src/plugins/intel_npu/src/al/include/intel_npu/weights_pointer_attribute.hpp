// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/runtime_attribute.hpp"

namespace intel_npu {

class WeightsPointerAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightsPointerAttribute", "0", RuntimeAttribute);

    WeightsPointerAttribute() = delete;

    WeightsPointerAttribute(const void* pointer) : memory_pointer(reinterpret_cast<size_t>(pointer)) {}

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("ptr", memory_pointer);
        return true;
    }

    size_t memory_pointer;
};

}  // namespace intel_npu
