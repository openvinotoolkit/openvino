// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass_config.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void PassConfig::disable(const DiscreteTypeInfo& type_info) {
    m_enabled.erase(type_info);
    m_disabled.insert(type_info);
}

void PassConfig::enable(const DiscreteTypeInfo& type_info) {
    m_enabled.insert(type_info);
    m_disabled.erase(type_info);
}

bool PassConfig::is_disabled(const DiscreteTypeInfo& type_info) const {
    return m_disabled.count(type_info);
}

bool PassConfig::is_enabled(const DiscreteTypeInfo& type_info) const {
    return m_enabled.count(type_info);
}

bool operator==(const PassConfig& lhs, const PassConfig& rhs) {
    return lhs.m_disabled == rhs.m_disabled && lhs.m_enabled == rhs.m_enabled;
}

bool operator!=(const PassConfig& lhs, const PassConfig& rhs) {
    return !(lhs == rhs);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
