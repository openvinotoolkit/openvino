#pragma once
#include <string>
#include <algorithm>

namespace ov::intel_gpu {

enum class LogLevel : int {
    NONE = 0,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    TRACE
};

inline LogLevel log_level_from_string(std::string v) {
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "none") return LogLevel::NONE;
    if (v == "error") return LogLevel::ERROR;
    if (v == "warning") return LogLevel::WARNING;
    if (v == "info") return LogLevel::INFO;
    if (v == "debug") return LogLevel::DEBUG;
    if (v == "trace") return LogLevel::TRACE;
    return LogLevel::ERROR;
}

} // namespace ov::intel_gpu
