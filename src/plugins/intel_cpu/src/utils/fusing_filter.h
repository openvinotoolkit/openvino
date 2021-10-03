// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#ifdef CPU_DEBUG_CAPS
#include "utils/debug_caps_config.h"
#include "config.h"
#include "node.h"

namespace ov {
namespace intel_cpu {

bool isFusingDisabled(const NodePtr& fuser, const NodePtr& fusee, const DebugCapsConfig& config);

} // namespace intel_cpu
} // namespace ov
#endif // CPU_DEBUG_CAPS
