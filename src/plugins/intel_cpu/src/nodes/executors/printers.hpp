// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/eltwise_config.hpp"
#ifdef CPU_DEBUG_CAPS
#    pragma once

#    include <ostream>

#    include "executor_config.hpp"

namespace ov::intel_cpu {

namespace executor {
template <typename Attrs>
struct Config;
}

struct FCAttrs;
struct ConvAttrs;

std::ostream& operator<<(std::ostream& os, const FCAttrs& attrs);
std::ostream& operator<<(std::ostream& os, const ConvAttrs& attrs);
std::ostream& operator<<(std::ostream& os, const EltwiseAttrs& attrs);

template <typename Attrs>
std::ostream& operator<<(std::ostream& os, const executor::Config<Attrs>& config) {
    for (const auto& desc : config.descs) {
        const auto id = desc.first;
        const auto descPtr = desc.second;
        os << "[" << id << "]" << *descPtr << ";";
    }

    os << config.attrs;

    return os;
}

}  // namespace ov::intel_cpu

#endif  // CPU_DEBUG_CAPS
