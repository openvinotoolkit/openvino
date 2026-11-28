// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decision_log.hpp"

#include <sstream>

namespace ov {
namespace auto_plugin {

std::string DecisionLog::summary(const InstanceStatsSnapshot& stats) const {
    std::ostringstream out;
    out << "total inference:" << inferences() << ", device switch:" << switches();
    for (const auto& [instance, latency] : stats.instances) {
        out << "\n  " << instance.to_string() << ": inference:" << latency.total
            << ", baseline:" << latency.baseline.count() << " us, recent:" << latency.recent.count() << " us";
    }
    return out.str();
}

}  // namespace auto_plugin
}  // namespace ov
