// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_gpu/tp_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace tp_gpu {

std::vector<std::string> TPConfig::requested_device_names() const {
    if (get_tp_size() == 0 && get_device_ids().empty()) {
        return {};
    }
    return resolve_device_names();
}

std::vector<std::string> TPConfig::resolve_device_names() const {
    std::vector<std::string> names = get_device_ids();
    const uint32_t size = get_tp_size();

    OPENVINO_ASSERT(size != 0 || !names.empty(),
                    "[TP_GPU] Neither ",
                    tp_size.name(),
                    " nor ",
                    device_ids.name(),
                    " was set, at least one of them must be specified");

    const std::size_t count = size != 0 ? static_cast<std::size_t>(size) : names.size();

    if (names.empty()) {
        names.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            names.push_back("GPU." + std::to_string(i));
        }
    } else {
        OPENVINO_ASSERT(names.size() == count,
                        "[TP_GPU] ",
                        device_ids.name(),
                        " lists ",
                        names.size(),
                        " devices, which does not match ",
                        tp_size.name(),
                        "=",
                        count);
    }

    OPENVINO_ASSERT(names.size() >= 2, "[TP_GPU] Need at least 2 devices, got ", names.size());
    return names;
}

bool TPConfig::owns(const std::string& name) const {
    return m_options_map.find(name) != m_options_map.end();
}

void TPConfig::erase_own_properties(ov::AnyMap& config) const {
    for (const auto& entry : m_options_map) {
        config.erase(entry.first);
    }
}

std::vector<ov::PropertyName> TPConfig::supported_properties() const {
    std::vector<ov::PropertyName> properties;
    for (const auto& entry : m_options_map) {
        if (entry.second->get_visibility() == ov::OptionVisibility::RELEASE) {
            properties.emplace_back(entry.first, ov::PropertyMutability::RW);
        }
    }
    return properties;
}

}  // namespace tp_gpu
}  // namespace ov
