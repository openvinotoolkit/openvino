// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pass_config.hpp"

ov::pass::PassConfig::PassConfig() {
    m_callback = [](const std::shared_ptr<const ::ov::Node>&) {
        return false;
    };
}

ov::pass::param_callback ov::pass::PassConfig::get_callback(const DiscreteTypeInfo& type_info) const {
    const auto& it = m_callback_map.find(type_info);
    if (it != m_callback_map.end()) {
        return it->second;
    } else {
        return m_callback;
    }
}

void ov::pass::PassConfig::enable(const ov::DiscreteTypeInfo& type_info) {
    m_disabled.erase(type_info);
    m_enabled.insert(type_info);
}

void ov::pass::PassConfig::disable(const ov::DiscreteTypeInfo& type_info) {
    m_enabled.erase(type_info);
    m_disabled.insert(type_info);
}

void ov::pass::PassConfig::add_disabled_passes(const PassConfig& rhs) {
    for (const auto& pass : rhs.m_disabled) {
        if (is_enabled(pass))
            continue;
        disable(pass);
    }
}
