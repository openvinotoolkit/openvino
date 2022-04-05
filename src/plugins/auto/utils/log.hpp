// Copyright (C) 2018-2022 Intel Corporation
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
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>

#include "singleton.hpp"
#include "time_utils.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

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
    return parseInteger(std::getenv("OPENVINO_LOG_LEVEL"));
}
const int debug_level = getDebugLevel();
enum class LogLevel : uint32_t {
    FREQUENT = 0x01,
    PROCESS = 0x02,
    DEBUG = 0x04,
    INFO = 0x08,
    WARN = 0x10,
    ERROR = 0x40,
    FATAL = 0x80,
    LOG_NONE = 0,
    LOG_FATAL = static_cast<uint32_t>(LogLevel::FATAL),
    LOG_ERROR = static_cast<uint32_t>(LogLevel::ERROR) | static_cast<uint32_t>(LogLevel::LOG_FATAL),
    LOG_WARNING = static_cast<uint32_t>(LogLevel::WARN) | static_cast<uint32_t>(LogLevel::LOG_ERROR),
    LOG_INFO = static_cast<uint32_t>(LogLevel::INFO) | static_cast<uint32_t>(LogLevel::LOG_WARNING),
    LOG_DEBUG = static_cast<uint32_t>(LogLevel::DEBUG) | static_cast<uint32_t>(LogLevel::LOG_INFO),
    LOG_TRACE = static_cast<uint32_t>(LogLevel::PROCESS) | static_cast<uint32_t>(LogLevel::LOG_DEBUG),
    LOG_FREQUENT = static_cast<uint32_t>(LogLevel::FREQUENT) | static_cast<uint32_t>(LogLevel::LOG_TRACE)
};

using LogTask = std::function<void()>;

class Log : public Singleton<Log> {
public:
    void setPrefix(std::string prefix);
    void setSuffix(std::string suffix);
    void setLogLevel(LogLevel logLevel);

    template <typename... Args>
    void doLog(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
        const char* func, long line, const char* tag, const char* fmt, Args... args);
    void doRun(LogLevel level, const LogTask& task);
#ifdef MULTIUNITTEST
    Log(std::string unittest):Log() {
    }
#endif

private:
    Log();
    friend Singleton<Log>;
    static std::string colorBegin(LogLevel logLevel);
    static std::string colorEnd(LogLevel logLevel);
    void checkFormat(const char* fmt);
    MOCKTESTMACRO void print(std::stringstream& stream);

private:
    std::mutex mutex;
    std::string logName;
    std::string logPath;
    std::string prefix;
    std::string suffix;
    uint32_t logLevel;
    static uint32_t defaultLogLevel;
    static std::vector<std::string> validFormat;
};

inline Log::Log()
    : logLevel(defaultLogLevel) {
    switch (debug_level) {
        case 0: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_NONE);
                    break;
                }
        case 1: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_FATAL);
                    break;
                }
        case 2: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_ERROR);
                    break;
                }
        case 3: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_WARNING);
                    break;
                }
        case 4: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_INFO);
                    break;
                }
        case 5: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_DEBUG);
                    break;
                }
        case 6: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_TRACE);
                    break;
                }
        case 7: {
                    logLevel = static_cast<uint32_t>(LogLevel::LOG_FREQUENT);
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

inline void Log::setLogLevel(LogLevel logLevel_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    logLevel = static_cast<uint32_t>(logLevel_);
}

inline void Log::print(std::stringstream& stream) {
    std::cout << stream.str() << std::endl;
}

inline void Log::checkFormat(const char* fmt) {
    const char* charIter = fmt;
    std::string fmtStr = "";
    bool  bCollectFmtStr = false;
    while (*charIter != '\0') {
        if (bCollectFmtStr) {
            fmtStr += *charIter;
            switch (fmtStr.size()) {
                case 1:
                case 2:
                    {
                        auto iter = std::find(validFormat.begin(), validFormat.end(), fmtStr);
                        if (iter != validFormat.end()) {
                            bCollectFmtStr = false;
                            fmtStr = "";
                        }
                        break;
                    }
                default:
                    {
                        throw std::runtime_error("format %" + fmtStr + " is not valid in log");
                        break;
                    }
            }
            charIter++;
            continue;
        }

        if (*charIter == '%') {
            bCollectFmtStr = true;
        }
        charIter++;
    }
    if (bCollectFmtStr) {
        throw std::runtime_error("format %" + fmtStr + " is not valid in log");
    }
}

template <typename... Args>
inline void Log::doLog(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
    const char* func, const long line, const char* tag, const char* fmt, Args... args) {

    if (!(static_cast<uint32_t>(level) & static_cast<uint32_t>(logLevel)) || !on) {
        return;
    }

    std::stringstream stream;
    stream << colorBegin(level) << prefix << '[' << TimeUtils::getCurrentTime() << ']';

    if (level < LogLevel::ERROR) {
        stream << levelStr[0];
    } else {
        stream << levelStr;
    }
    stream << '[' << getFileName(file) << ':' << line << ']';

    if (isTraceCallStack) {
        stream << '[' << func << '(' << ')' << ']';
    }
    if (tag) {
        stream << '[' << tag << ']';
    }
    char buffer[255];
    std::string compatibleString;

    try {
        checkFormat(fmt);
        compatibleString =  "%s" + std::string(fmt);
        std::snprintf(&buffer[0], sizeof(buffer), compatibleString.c_str(), "", args...);
        stream << ' ' << buffer << suffix << colorEnd(level);
    } catch (std::runtime_error& err) {
        stream << ' ' << err.what() << colorEnd(level);
    }

    std::lock_guard<std::mutex> autoLock(mutex);
    print(stream);
}

inline void Log::doRun(LogLevel level, const LogTask& task) {
    if (!(static_cast<uint32_t>(level) & static_cast<uint32_t>(logLevel))) {
        return;
    }
    task();
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
