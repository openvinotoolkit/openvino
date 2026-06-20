// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <utility>

#include "openvino/util/log.hpp"

namespace ov::hetero {

enum class PerfLogLevel {
    Disabled = 0,
    Basic = 1,
    SplitDetails = 2,
};

inline PerfLogLevel perf_log_level() {
    static const auto level = []() {
        const char* value = std::getenv("OPENVINO_HETERO_PERF");
        if (value == nullptr || *value == '\0') {
            return PerfLogLevel::Disabled;
        }

        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || (end != nullptr && *end != '\0')) {
            return PerfLogLevel::Disabled;
        }

        if (parsed <= 0) {
            return PerfLogLevel::Disabled;
        }

        if (parsed >= static_cast<long>(PerfLogLevel::SplitDetails)) {
            return PerfLogLevel::SplitDetails;
        }

        return PerfLogLevel::Basic;
    }();
    return level;
}

inline bool perf_log_enabled() {
    return perf_log_level() != PerfLogLevel::Disabled;
}

inline bool perf_log_enabled(PerfLogLevel level) {
    if (level == PerfLogLevel::Disabled) {
        return false;
    }
    return static_cast<int>(perf_log_level()) >= static_cast<int>(level);
}

template <typename... Args>
inline void perf_log_impl(const char* file, int line, PerfLogLevel level, Args&&... args) {
    if (!perf_log_enabled(level)) {
        return;
    }

    ov::util::LogHelper helper(ov::util::LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY, file, line);
    auto& stream = helper.stream();
    (stream << ... << std::forward<Args>(args));
}

}  // namespace ov::hetero

#define HETERO_PERF_LOG_LEVEL(level, ...) ::ov::hetero::perf_log_impl(__FILE__, __LINE__, level, __VA_ARGS__)
#define HETERO_PERF_LOG(...)              HETERO_PERF_LOG_LEVEL(::ov::hetero::PerfLogLevel::Basic, __VA_ARGS__)