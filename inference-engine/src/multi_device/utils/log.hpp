// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MULTIDEVICEPLUGIN_LOG_H
#define MULTIDEVICEPLUGIN_LOG_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>

#include "singleton.hpp"
#include "time_utils.hpp"
#include "thread_utils.hpp"

#ifdef COLOR_LOG
#define COL(x) "\033[1;" #x ";40m"
#define COL_END "\033[0m"
#else
#define COL(x) ""
#define COL_END ""
#endif

#define RED COL(31)
#define GREEN COL(32)
#define YELLOW COL(33)
#define BLUE COL(34)
#define MAGENTA COL(35)
#define CYAN COL(36)
#define WHITE COL(0)
#define DEFAULT_COLOR ""

#ifdef ERROR
#undef ERROR
#endif

namespace MultiDevicePlugin {
inline int parseInteger(const char* str) {
    std::string var(str ? str : "");
    try {
        return std::stoi(var);
    } catch (...) {
        return INT32_MAX;
    }
}

inline std::string getFileName(const std::string& filePath) {
    auto index = filePath.find_last_of("/\\");
    if (std::string::npos == index) {
        return filePath;
    }
    return filePath.substr(index + 1);
}

inline int getDebugLevel() {
    return parseInteger(std::getenv("DEBUG_MULTIPLUGIN_LEVEL"));
}
const int debug_level = getDebugLevel();

enum class LogLevel : uint32_t {
    FREQUENT = 0x01,
    PROCESS = 0x02,
    DEBUG = 0x04,
    INFO = 0x08,
    WARN = 0x10,
    ERROR = 0x40,
    FATAL = 0x80
};

class Log : public Singleton<Log> {
public:
    void setPrefix(std::string prefix);
    void setSuffix(std::string suffix);
    void setLogLevel(uint32_t logLevel);

    template <typename... Args>
    void doLog(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
        const char* func, long line, const char* tag, const char* fmt, Args... args);

private:
    Log();
    friend Singleton<Log>;
    static std::string colorBegin(LogLevel logLevel);
    static std::string colorEnd(LogLevel logLevel);

private:
    std::mutex mutex;
    std::string logName;
    std::string logPath;
    std::string prefix;
    std::string suffix;
    uint32_t logLevel;
    static uint32_t defaultLogLevel;
};

inline Log::Log()
    : logLevel(defaultLogLevel) {
    setPriority(0);
    switch (debug_level) {
        case 1: {
                    logLevel = static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 2: {
                    logLevel = static_cast<uint32_t>(LogLevel::ERROR) | static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 3: {
                    logLevel = static_cast<uint32_t>(LogLevel::WARN) |
                        static_cast<uint32_t>(LogLevel::ERROR) |
                        static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 4: {
                    logLevel = static_cast<uint32_t>(LogLevel::INFO) |
                        static_cast<uint32_t>(LogLevel::WARN) |
                        static_cast<uint32_t>(LogLevel::ERROR) |
                        static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 5: {
                    logLevel = static_cast<uint32_t>(LogLevel::DEBUG) |
                        static_cast<uint32_t>(LogLevel::INFO) |
                        static_cast<uint32_t>(LogLevel::WARN) |
                        static_cast<uint32_t>(LogLevel::ERROR) |
                        static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 6: {
                    logLevel = static_cast<uint32_t>(LogLevel::PROCESS) |
                        static_cast<uint32_t>(LogLevel::DEBUG) |
                        static_cast<uint32_t>(LogLevel::INFO) |
                        static_cast<uint32_t>(LogLevel::WARN) |
                        static_cast<uint32_t>(LogLevel::ERROR) |
                        static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        case 7: {
                    logLevel = static_cast<uint32_t>(LogLevel::FREQUENT) |
                        static_cast<uint32_t>(LogLevel::PROCESS) |
                        static_cast<uint32_t>(LogLevel::INFO) |
                        static_cast<uint32_t>(LogLevel::DEBUG) |
                        static_cast<uint32_t>(LogLevel::WARN) |
                        static_cast<uint32_t>(LogLevel::ERROR) |
                        static_cast<uint32_t>(LogLevel::FATAL);
                    break;
                }
        default:
                break;
    }
}

inline void Log::setPrefix(std::string prefix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    prefix = std::move(prefix_);
}

inline void Log::setSuffix(std::string suffix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    suffix = std::move(suffix_);
}

inline void Log::setLogLevel(uint32_t logLevel_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    logLevel = logLevel_;
}
template <typename... Args>
inline void Log::doLog(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
    const char* func, const long line, const char* tag, const char* fmt, Args... args) {

    if (!(static_cast<uint32_t>(level) & static_cast<uint32_t>(logLevel)) || !on) {
        return;
    }

    std::stringstream stream;
    stream << colorBegin(level) << prefix << '[' << TimeUtils::getCurrentTime() << ']';

#ifdef VERBOSE_LOG
    stream << "[" << ThreadUtils::getThreadId() << "][" << levelStr << "]["
           << getFileName(file) << ':' << func << ':' << line << ']';
#else
    stream << '[' << ThreadUtils::getThreadId() << ']';
    if (level < LogLevel::ERROR) {
        stream << levelStr[0];
    } else {
        stream << levelStr;
    }
    stream << '[' << getFileName(file) << ':' << line << ']';
#endif

    if (isTraceCallStack) {
        stream << '[' << func << '(' << ')' << ']';
    }
    if (tag) {
        stream << '[' << tag << ']';
    }
    char buffer[255];
    snprintf (&buffer[0], sizeof(buffer), fmt, args...);
    stream << ' ' << buffer << suffix << colorEnd(level);
    std::lock_guard<std::mutex> autoLock(mutex);
    std::cout << stream.str() << std::endl;
}

inline std::string Log::colorBegin(MultiDevicePlugin::LogLevel logLevel) {
    if (logLevel == LogLevel::WARN) {
        return std::string(CYAN);
    }
    if (logLevel == LogLevel::ERROR || logLevel == LogLevel::FATAL) {
        return std::string(RED);
    }
    return std::string(DEFAULT_COLOR);
}

inline std::string Log::colorEnd(MultiDevicePlugin::LogLevel logLevel) {
    if (logLevel == LogLevel::WARN || logLevel == LogLevel::ERROR || logLevel == LogLevel::FATAL) {
        return std::string(COL_END);
    }
    return {};
}
} // namespace MultiDevicePlugin

#endif //MULTIDEVICEPLUGIN_LOG_H
