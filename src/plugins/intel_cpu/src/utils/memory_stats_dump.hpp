// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef CPU_DEBUG_CAPS
#    include "compiled_model.h"
#    include "weights_cache.hpp"

namespace ov::intel_cpu {
void dumpMemoryStats(const DebugCapsConfig& conf,
                     const std::string& network_name,
                     std::deque<CompiledModel::GraphGuard>& graphs,
                     const SocketsWeights& weights_cache);
}  // namespace ov::intel_cpu
#endif  // CPU_DEBUG_CAPS