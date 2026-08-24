// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "openvino/runtime/plugin_config.hpp"

namespace ov::intel_gpu::cache {

class ConfigImportAttributeVisitor final : public ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer> {
public:
    explicit ConfigImportAttributeVisitor(cldnn::BinaryInputBuffer& input) : ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer>(input), m_input(input) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        static const auto encryption_callbacks_property = std::string{ov::cache_encryption_callbacks.name()};
        static const auto encryption_callbacks_attribute = encryption_callbacks_property + "__internal";
        if (name == encryption_callbacks_attribute) {
            // Function objects are process-local and cannot be reconstructed from a cache blob. Consume the
            // serialized placeholder and preserve callbacks supplied to the current import_model() call.
            std::string ignored;
            m_input >> ignored;
            return;
        }
        ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer>::on_adapter(name, adapter);
    }

private:
    cldnn::BinaryInputBuffer& m_input;
};

}  // namespace ov::intel_gpu::cache
