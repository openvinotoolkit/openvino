// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/logger.hpp"
#include "npu_bytecode_utils/format.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace ov::intel_npu::vm;

namespace {

constexpr std::string_view ANSI_RESET = "\x1b[0m";
constexpr std::string_view ANSI_MAGENTA_BOLD = "\033[1;35m";
constexpr std::string_view ANSI_RED_BOLD = "\033[1;31m";
constexpr std::string_view ANSI_YELLOW_BOLD = "\033[1;33m";
constexpr std::string_view ANSI_BLUE = "\033[0;34m";
constexpr std::string_view ANSI_GREEN = "\033[0;32m";
constexpr std::string_view ANSI_CYAN = "\033[0;36m";
constexpr std::string_view ANSI_WHITE = "\033[0;37m";

#if defined(VPUX_DEVELOPER_BUILD)

std::optional<std::string_view> isEnvVarSet(std::string_view envVar) {
    const auto env = std::getenv(envVar.data());
    if (env != nullptr && *env != '\0') {
        return std::string_view(env);
    }
    return std::nullopt;
}

struct LogFilterState {
    std::optional<std::regex> compiledRegex;
    bool hasFilter{false};
};

// Cached values for environment variables to avoid performance overhead of getenv() and regex matching in hot code
// paths. These are mutable globals but are set once. They are not made constant for unit tests, as they need to be able
// to reset these globals in between tests.
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
static std::once_flag filterStateFlag;
static std::once_flag areColorsActiveFlag;
static std::once_flag showFullFilePathFlag;
static std::once_flag logLevelFlag;
static std::optional<LogFilterState> cachedFilterState;
static std::optional<bool> cachedAreColorsActive;
static std::optional<bool> cachedShowFullFilePath;
static std::optional<LogLevel> cachedLogLevel;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

const LogFilterState& getLogFilterState() {
    // Cache once to avoid getenv() and regex compilation overhead in hot logging call sites.
    // std::call_once ensures thread-safe one-time initialization with minimal overhead on reads.
    std::call_once(filterStateFlag, []() {
        std::optional<std::string_view> filter;
        if (auto logFilterStr = isEnvVarSet("NPU_VM_LOG_FILTER"); logFilterStr.has_value()) {
            filter = logFilterStr;
        } else if (auto logFilterStr = isEnvVarSet("IE_NPU_LOG_FILTER"); logFilterStr.has_value()) {
            filter = logFilterStr;
        }

        LogFilterState state;
        if (filter.has_value()) {
            state.hasFilter = true;
            try {
                // std::regex can throw if the provided filter is not a valid regex. In that case, we consider all logs
                // as filtered out
                state.compiledRegex.emplace(filter.value().data());
            } catch (const std::regex_error&) {
                // Invalid regex disables all filtered logs.
                state.compiledRegex.reset();
            }
        }

        cachedFilterState.emplace(state);
    });
    return cachedFilterState.value();
}

LogLevel parseLogLevelString(std::string_view logLevelStr) {
    if (logLevelStr == "LOG_TRACE") {
        return LogLevel::Trace;
    }
    if (logLevelStr == "LOG_DEBUG") {
        return LogLevel::Debug;
    }
    if (logLevelStr == "LOG_INFO") {
        return LogLevel::Info;
    }
    if (logLevelStr == "LOG_WARN") {
        return LogLevel::Warn;
    }
    if (logLevelStr == "LOG_ERROR") {
        return LogLevel::Error;
    }
    if (logLevelStr == "LOG_FATAL") {
        return LogLevel::Fatal;
    }
    return LogLevel::None;
}

#endif  // defined(VPUX_DEVELOPER_BUILD)

// Check whether terminal colors are enabled. By default, colors are enabled. For developer builds, the user can disable
// colors by setting the NPU_VM_LOG_COLORS environment variable to "0"
bool areColorsActive() {
#if defined(VPUX_DEVELOPER_BUILD)
    // std::call_once ensures thread-safe one-time initialization with minimal overhead on reads.
    std::call_once(areColorsActiveFlag, []() {
        bool value = true;
        if (const auto logColorsStr = isEnvVarSet("NPU_VM_LOG_COLORS"); logColorsStr.has_value()) {
            value = logColorsStr.value() != "0";
        }
        cachedAreColorsActive.emplace(value);
    });
    return cachedAreColorsActive.value();
#endif
    return true;
}

// Check whether the logs should include the full file path or just the file name. By default, only the file name is
// shown. For developer builds, the user can enable full file paths by setting the NPU_VM_LOG_FULL_PATH environment
// variable to a value that is not "0"
bool showFullFilePath() {
#if defined(VPUX_DEVELOPER_BUILD)
    // std::call_once ensures thread-safe one-time initialization with minimal overhead on reads.
    std::call_once(showFullFilePathFlag, []() {
        bool value = false;
        if (const auto logFullPathStr = isEnvVarSet("NPU_VM_LOG_FULL_PATH"); logFullPathStr.has_value()) {
            value = logFullPathStr.value() != "0";
        }
        cachedShowFullFilePath.emplace(value);
    });
    return cachedShowFullFilePath.value();
#endif
    return false;
}

// Extracts the file name from a full file path, or returns the full path based on the showFullFilePath() setting
std::string filePath(const char* file) {
    if (file == nullptr) {
        return {};
    }
    std::string filePath(file);
    if (showFullFilePath()) {
        return filePath;
    }
    const auto separatorPos = filePath.find_last_of("/\\");
    if (separatorPos == std::string::npos) {
        return filePath;
    }
    return filePath.substr(separatorPos + 1);
}

std::string getLogSource(const char* file, int line) {
    return filePath(file) + ":" + std::to_string(line);
}

const char* levelPrefix(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return "[TRCE]";
    case LogLevel::Debug:
        return "[DEBG]";
    case LogLevel::Info:
        return "[INFO]";
    case LogLevel::Warn:
        return "[WARN]";
    case LogLevel::Error:
        return "[ERRO]";
    case LogLevel::Fatal:
        return "[FATL]";
    default:
        return "[UNKN]";
    }
}

std::string_view color(std::string_view colorCode) {
    if (areColorsActive()) {
        return colorCode;
    }
    return {};
}

std::string_view colorLevel(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return color(ANSI_CYAN);
    case LogLevel::Debug:
        return color(ANSI_GREEN);
    case LogLevel::Info:
        return color(ANSI_BLUE);
    case LogLevel::Warn:
        return color(ANSI_YELLOW_BOLD);
    case LogLevel::Error:
        return color(ANSI_RED_BOLD);
    case LogLevel::Fatal:
        return color(ANSI_MAGENTA_BOLD);
    default:
        return {};
    }
}

// Generate a timestamp string in the format "YYYY-MM-DD HH:MM:SS.mmm"
std::string makeTimestamp() {
    const auto localTimeFromTimeT = [](std::time_t secondsSinceEpoch) -> std::tm {
        std::tm nowTm{};
#if defined(_WIN32)
        localtime_s(&nowTm, &secondsSinceEpoch);
#else
        localtime_r(&secondsSinceEpoch, &nowTm);
#endif
        return nowTm;
    };

    const auto now = std::chrono::system_clock::now();
    const auto secondsTimePoint = std::chrono::time_point_cast<std::chrono::seconds>(now);
    const auto secondsSinceEpoch = std::chrono::system_clock::to_time_t(secondsTimePoint);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now - secondsTimePoint).count();
    const auto nowTm = localTimeFromTimeT(secondsSinceEpoch);

    std::ostringstream stream;
    stream << std::put_time(&nowTm, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << millis;
    return stream.str();
}

}  // namespace

#if defined(VPUX_DEVELOPER_BUILD)

void intel_npu::vm::detail::resetLoggingCache() {
    // Reset cached values and reinitialize once_flags for testing.
    // Note: std::once_flag cannot be reset, so we destroy and recreate them using placement new.
    cachedLogLevel.reset();
    cachedAreColorsActive.reset();
    cachedShowFullFilePath.reset();
    cachedFilterState.reset();
    logLevelFlag.~once_flag();
    new (&logLevelFlag) std::once_flag();
    areColorsActiveFlag.~once_flag();
    new (&areColorsActiveFlag) std::once_flag();
    showFullFilePathFlag.~once_flag();
    new (&showFullFilePathFlag) std::once_flag();
    filterStateFlag.~once_flag();
    new (&filterStateFlag) std::once_flag();
}

#endif  // defined(VPUX_DEVELOPER_BUILD)

// For developer builds, the effective log level can be set by the user through the NPU_VM_LOG_LEVEL or OV_NPU_LOG_LEVEL
// environment variables. For non-developer builds, the log level is Info by default
LogLevel ov::intel_npu::vm::detail::getEffectiveLogLevel() {
#if !defined(VPUX_DEVELOPER_BUILD)
    return LogLevel::Info;
#else
    // std::call_once ensures thread-safe one-time initialization with minimal overhead on reads.
    std::call_once(logLevelFlag, []() {
        LogLevel value = LogLevel::Info;
        if (auto logLevelStr = isEnvVarSet("NPU_VM_LOG_LEVEL"); logLevelStr.has_value()) {
            value = parseLogLevelString(logLevelStr.value());
        } else if (auto logLevelStr = isEnvVarSet("OV_NPU_LOG_LEVEL"); logLevelStr.has_value()) {
            value = parseLogLevelString(logLevelStr.value());
        }
        cachedLogLevel.emplace(value);
    });
    return cachedLogLevel.value();
#endif
}

// Check whether the log message should be printed based on the log filter. Only developer builds support log filtering,
// which is done by matching the log source (file name / path and line) against a regex specified in the
// NPU_VM_LOG_FILTER or IE_NPU_LOG_FILTER environment variables. For non-developer builds, all messages are printed
bool ov::intel_npu::vm::detail::isLogIncludedInFilter([[maybe_unused]] const char* file, [[maybe_unused]] int line) {
#if !defined(VPUX_DEVELOPER_BUILD)
    return true;
#else
    const auto& filterState = getLogFilterState();
    if (!filterState.hasFilter) {
        return true;
    }
    if (!filterState.compiledRegex.has_value()) {
        return false;
    }
    const auto logSource = getLogSource(file, line);
    return std::regex_search(logSource.begin(), logSource.end(), *filterState.compiledRegex);
#endif
}

// Print the log message to the appropriate stream with the specified formatting and colors based on the log level
void ov::intel_npu::vm::detail::dispatchLogMessage(LogLevel level, const char* file, int line, std::string_view format,
                                               const std::vector<std::string>& args) {
    static std::mutex logMutex;
    std::lock_guard<std::mutex> lock(logMutex);
    std::ostringstream stream;
    stream << color(ANSI_WHITE) << "[" << makeTimestamp() << "] " << colorLevel(level) << levelPrefix(level) << ' '
           << getLogSource(file, line) << color(ANSI_RESET) << ' ' << formatString(format, args) << '\n';
    std::cerr << stream.str();
}
