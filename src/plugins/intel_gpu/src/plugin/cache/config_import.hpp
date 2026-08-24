// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/plugin_config.hpp"

namespace ov::intel_gpu::cache {

class ConfigImportAttributeVisitor final : public ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer> {
public:
    explicit ConfigImportAttributeVisitor(cldnn::BinaryInputBuffer& input) : ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer>(input), m_input(input) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        static const std::array<std::string, 3> process_local_attributes = {
            serialized_attribute_name(ov::cache_encryption_callbacks.name()),
            serialized_attribute_name(ov::hint::model.name()),
            serialized_attribute_name(ov::intel_gpu::weightless_attr.name()),
        };
        if (std::find(process_local_attributes.begin(), process_local_attributes.end(), name) != process_local_attributes.end()) {
            // Function objects and shared ownership are process-local and cannot be reconstructed from a cache blob.
            // Consume the serialized placeholder and preserve values supplied to the current import_model() call.
            std::string ignored;
            m_input >> ignored;
            return;
        }
        ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer>::on_adapter(name, adapter);
    }

private:
    static std::string serialized_attribute_name(std::string_view property_name) {
        return std::string{property_name} + "__internal";
    }

    cldnn::BinaryInputBuffer& m_input;
};

}  // namespace ov::intel_gpu::cache
