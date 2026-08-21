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
        if (auto option_adapter = ov::as_type<ov::AttributeAdapter<ov::ConfigOptionBase*>>(&adapter)) {
            auto* option = option_adapter->get();
            if (option->get_visibility() == ov::OptionVisibility::RELEASE || option->get_visibility() == ov::OptionVisibility::RELEASE_INTERNAL) {
                std::string serialized_value;
                m_input >> serialized_value;
                try {
                    if (option->is_valid_value(serialized_value)) {
                        option->set_any(serialized_value);
                    }
                } catch (...) {
                    // Some GPU options contain process-local functions or plugin-defined aggregate types that
                    // cannot cross a shared-library RTTI boundary on Android. Keep the current import config,
                    // matching the base visitor's policy of ignoring values that cannot be reconstructed.
                }
            }
            return;
        }
        ov::IstreamAttributeVisitor<cldnn::BinaryInputBuffer>::on_adapter(name, adapter);
    }

private:
    cldnn::BinaryInputBuffer& m_input;
};

}  // namespace ov::intel_gpu::cache
