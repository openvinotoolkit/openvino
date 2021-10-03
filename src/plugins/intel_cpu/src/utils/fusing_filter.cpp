// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "utils/debug_caps_config.h"
#ifdef CPU_DEBUG_CAPS

#include <unordered_map>
#include <iostream>

#include "fusing_filter.h"
#include "config.h"
#include "node.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

bool isFusingDisabled(const NodePtr& target, const NodePtr& fused, const DebugCapsConfig& config) {
    const auto& byTargetName = config.disable.fusingFilterByTargetName.filter;
    const auto& byFusedName = config.disable.fusingFilterByFusedName.filter;
    const auto& byTargetType = config.disable.fusingFilterByTargetType.filter;
    const auto& byFusedTupe = config.disable.fusingFilterByFusedType.filter;

    if (byTargetName.count(target->getName()) || byTargetName.count("*") ||
        byFusedName.count(fused->getName())   || byFusedName.count("*")  ||
        byTargetType.count(NameFromType(target->getType()))              ||
        byFusedTupe.count(NameFromType(fused->getType())))
        return true;

    return false;
}

} // namespace intel_cpu
} // namespace ov

#endif // CPU_DEBUG_CAPS
