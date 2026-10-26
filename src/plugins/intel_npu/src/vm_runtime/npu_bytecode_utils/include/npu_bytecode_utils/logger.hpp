//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
// Logging utility for NPU bytecode reader and related components. Provides macros for logging at different levels
// (Trace, Debug, Info, Warn, Error, Fatal) with support for formatted messages and color output. The log messages
// include a timestamp, log level, and source information (file name / path and line number):
//   [YYYY-MM-DD HH:MM:SS.mmm] [LOG_LEVEL] file:line message
//
// All logs are printed to standard error.
//
// Logs can be disabled during build by setting the BUILD_LOG_LEVEL build option to a value from 0 (no logs) to 6 (all
// logs). For example, setting BUILD_LOG_LEVEL to 4 will include Info, Warn, Error, and Fatal messages, but exclude
// Debug and Trace messages.
//
// For developer builds (i.e. when VPUX_DEVELOPER_BUILD is defined), the logging behavior can be further customized
// through environment variables:
// - NPU_VM_LOG_LEVEL or OV_NPU_LOG_LEVEL: Set the log level threshold (e.g. LOG_TRACE, LOG_DEBUG, LOG_INFO, LOG_WARN,
// LOG_ERROR, LOG_FATAL). If both are set, NPU_VM_LOG_LEVEL takes precedence.
// - NPU_VM_LOG_FILTER or IE_NPU_LOG_FILTER: A regex pattern to filter log messages based on their source (file name /
// path and line). Only messages whose source matches the regex will be printed. If both are set, NPU_VM_LOG_FILTER
// takes precedence.
// - NPU_VM_LOG_COLORS: Enable or disable colored output. If set to "0", colors are disabled. By default, colors are
// enabled.
// - NPU_VM_LOG_FULL_PATH: Control whether log messages include the full file path or just the file name. If set to "0",
// only the file name is included. By default, only the file name is shown.
//
// For non-developer builds, the log level is set to Info by default, colors are enabled, only the file name is printed
// and all messages are included regardless of their source.

namespace intel_npu::vm {

enum class LogLevel : uint8_t {
    None = 0,
    Fatal = 1,
    Error = 2,
    Warn = 3,
    Info = 4,
    Debug = 5,
    Trace = 6,
};

namespace detail {

#if defined(VPUX_DEVELOPER_BUILD)
// Reset all logging caches. Intended for unit test isolation; not thread-safe and must not be called concurrently with
// logging
void resetLoggingCache();
#endif

LogLevel getEffectiveLogLevel();

bool isLogIncludedInFilter(const char* file, int line);
inline bool isLogLevelActive(LogLevel level) {
    return level <= getEffectiveLogLevel();
}

void dispatchLogMessage(LogLevel level, const char* file, int line, std::string_view format,
                        const std::vector<std::string>& args);

inline void dispatchLog(LogLevel level, const char* file, int line, std::string_view format) {
    try {
        if (!isLogIncludedInFilter(file, line)) {
            return;
        }
        const std::vector<std::string> args;
        dispatchLogMessage(level, file, line, format, args);
    } catch (const std::exception& e) {
        try {
            std::cerr << "Exception during log printing: " << e.what() << std::endl;
        } catch (...) {
            // Swallow exceptions from fallback logging to keep logging noexcept.
        }
    } catch (...) {
        // Swallow any non-std exception to keep logging noexcept.
    }
}

template <typename... Args>
void dispatchLog(LogLevel level, const char* file, int line, std::string_view format, const Args&... args) {
    try {
        const auto toLogString = [](const auto& value) -> std::string {
            using ValueType = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<ValueType, std::string>) {
                return value;
            } else if constexpr (std::is_same_v<ValueType, std::string_view>) {
                return std::string(value);
            } else if constexpr (std::is_same_v<ValueType, const char*> || std::is_same_v<ValueType, char*>) {
                const char* strValue = value;  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
                return strValue == nullptr ? std::string{} : std::string(strValue);
            } else {
                std::ostringstream stream;
                stream << value;
                return stream.str();
            }
        };
        if (!isLogIncludedInFilter(file, line)) {
            return;
        }
        const std::vector<std::string> serializedArgs{toLogString(args)...};
        dispatchLogMessage(level, file, line, format, serializedArgs);
    } catch (const std::exception& e) {
        try {
            std::cerr << "Exception during log printing: " << e.what() << std::endl;
        } catch (...) {
            // Swallow exceptions from fallback logging to keep logging noexcept.
        }
    } catch (...) {
        // Swallow any non-std exception to keep logging noexcept.
    }
}

}  // namespace detail

}  // namespace intel_npu::vm

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#define NPU_VM_LOG_DETAIL_IMPL(level, ...)                                                                           \
    do { /* NOLINT(cppcoreguidelines-avoid-do-while) */                                                              \
        if (::intel_npu::vm::detail::isLogLevelActive(::intel_npu::vm::LogLevel::level)) {                           \
            ::intel_npu::vm::detail::dispatchLog(::intel_npu::vm::LogLevel::level, __FILE__, __LINE__, __VA_ARGS__); \
        }                                                                                                            \
    } while (0)

#if BUILD_LOG_LEVEL == 6
#define NPU_VM_LOG_TRACE(...) NPU_VM_LOG_DETAIL_IMPL(Trace, __VA_ARGS__)
#else
#define NPU_VM_LOG_TRACE(...) ((void)0)
#endif

#if BUILD_LOG_LEVEL >= 5
#define NPU_VM_LOG_DEBUG(...) NPU_VM_LOG_DETAIL_IMPL(Debug, __VA_ARGS__)
#else
#define NPU_VM_LOG_DEBUG(...) ((void)0)
#endif

#if BUILD_LOG_LEVEL >= 4
#define NPU_VM_LOG_INFO(...) NPU_VM_LOG_DETAIL_IMPL(Info, __VA_ARGS__)
#else
#define NPU_VM_LOG_INFO(...) ((void)0)
#endif

#if BUILD_LOG_LEVEL >= 3
#define NPU_VM_LOG_WARN(...) NPU_VM_LOG_DETAIL_IMPL(Warn, __VA_ARGS__)
#else
#define NPU_VM_LOG_WARN(...) ((void)0)
#endif

#if BUILD_LOG_LEVEL >= 2
#define NPU_VM_LOG_ERROR(...) NPU_VM_LOG_DETAIL_IMPL(Error, __VA_ARGS__)
#else
#define NPU_VM_LOG_ERROR(...) ((void)0)
#endif

#if BUILD_LOG_LEVEL >= 1
#define NPU_VM_LOG_FATAL(...) NPU_VM_LOG_DETAIL_IMPL(Fatal, __VA_ARGS__)
#else
#define NPU_VM_LOG_FATAL(...) ((void)0)
#endif

// NOLINTEND(cppcoreguidelines-macro-usage)
