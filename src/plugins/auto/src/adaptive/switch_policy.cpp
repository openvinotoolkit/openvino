// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "switch_policy.hpp"

namespace ov {
namespace auto_plugin {

SwitchDecision SwitchPolicy::decide(const SwitchInput& input) const {
    if (input.current.empty() || input.proposed == input.current) {
        return {true, "unchanged"};
    }
    if (!input.current_available) {
        return {false, "current device is not available"};
    }
    if (input.current_over_threshold) {
        return {false, "current device is over its threshold"};
    }
    if (input.now - input.current_since < m_config.min_dwell) {
        return {true, "minimum dwell time"};
    }
    return {false, "preferred device changed"};
}

}  // namespace auto_plugin
}  // namespace ov
