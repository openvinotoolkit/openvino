// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS
#pragma once

#include <ostream>
#include "executor_config.hpp"
#include "gemm_attrs.hpp"

namespace ov {
namespace intel_cpu {

namespace executor {
template<typename Attrs> struct Config;
}

struct FCAttrs;

std::ostream & operator<<(std::ostream & os, const FCAttrs& attrs);
std::ostream & operator<<(std::ostream & os, const GEMMAttrs& attrs);
std::ostream & operator<<(std::ostream & os, const PostOps& postOps);

template<typename Attrs>
std::ostream & operator<<(std::ostream & os, const executor::Config<Attrs>& config) {
    for (const auto& desc : config.descs) {
        const auto id = desc.first;
        const auto descPtr = desc.second;
        os << "[" << id << "]" << *descPtr << ";";
    }

    os << config.postOps;
    os << config.attrs;

    return os;
}

}   // namespace intel_cpu
}   // namespace ov

#endif // CPU_DEBUG_CAPS
