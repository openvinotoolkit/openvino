// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "perf.hpp"

#include <cstdlib>
#include <string>

#include "logging.hpp"

std::size_t ov::npuw::perf::keep_runs() {
    static const std::size_t runs = []() -> std::size_t {
        constexpr std::size_t default_runs = 8u;
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
        const auto* opt = std::getenv("OPENVINO_NPUW_PROF_KEEP_RUNS");
        if (opt == nullptr) {
            return default_runs;
        }
        try {
            const auto parsed = std::stoll(std::string(opt));
            return parsed <= 0 ? std::size_t{0u} : static_cast<std::size_t>(parsed);
        } catch (...) {
            LOG_WARN("NPUW profiling: bad OPENVINO_NPUW_PROF_KEEP_RUNS value, using " << default_runs);
        }
#endif
        return default_runs;
    }();
    return runs;
}
