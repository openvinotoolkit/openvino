// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass_config.hpp"

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"

namespace ov::snippets::lowered::pass {

void PassConfig::disable(const DiscreteTypeInfo& type_info) {
    m_enabled.erase(type_info);
    m_disabled.insert(type_info);
}

void PassConfig::enable(const DiscreteTypeInfo& type_info) {
    m_enabled.insert(type_info);
    m_disabled.erase(type_info);
}

bool PassConfig::is_disabled(const DiscreteTypeInfo& type_info) const {
    return m_disabled.count(type_info) != 0U;
}

bool PassConfig::is_enabled(const DiscreteTypeInfo& type_info) const {
    return m_enabled.count(type_info) != 0U;
}

PassConfig PassConfig::merge_configs(const PassConfig& lhs, const PassConfig& rhs) {
    PassConfig merged_config(lhs);
    merged_config.m_disabled.insert(rhs.m_disabled.begin(), rhs.m_disabled.end());
    merged_config.m_enabled.insert(rhs.m_enabled.begin(), rhs.m_enabled.end());
    for (const auto& type_info : merged_config.m_disabled) {
        OPENVINO_ASSERT(!merged_config.is_enabled(type_info),
                        "PassConfigs with conflicting settings for pass ",
                        type_info,
                        " cannot be merged.");
    }
    return merged_config;
}

bool operator==(const PassConfig& lhs, const PassConfig& rhs) {
    return lhs.m_disabled == rhs.m_disabled && lhs.m_enabled == rhs.m_enabled;
}

bool operator!=(const PassConfig& lhs, const PassConfig& rhs) {
    return !(lhs == rhs);
}

}  // namespace ov::snippets::lowered::pass
