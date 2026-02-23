#pragma once
#include <string>
#include <iostream>

namespace ov::intel_gpu {

enum class LogLevel {
    NONE = 0,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    TRACE
};

// Convert LogLevel to string for printing/config
static inline std::string to_string(LogLevel level) {
    switch (level) {
        case LogLevel::NONE:    return "NONE";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::TRACE:   return "TRACE";
    }
    return "UNKNOWN";
}

// For parsing from config string
static inline LogLevel parse_log_level(const std::string& s) {
    if (s == "ERROR")   return LogLevel::ERROR;
    if (s == "WARNING") return LogLevel::WARNING;
    if (s == "INFO")    return LogLevel::INFO;
    if (s == "DEBUG")   return LogLevel::DEBUG;
    if (s == "TRACE")   return LogLevel::TRACE;
    return LogLevel::NONE;
}

// Plugin property
static constexpr ov::Property<LogLevel> log_level {
    "LOG_LEVEL",
};

} // namespace ov::intel_gpu
