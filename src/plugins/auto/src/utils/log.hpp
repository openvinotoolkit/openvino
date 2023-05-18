// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef AUTOPLUGIN_LOG_H
#define AUTOPLUGIN_LOG_H

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
#define auto_plugin mock_auto_plugin
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
namespace ov {
namespace auto_plugin {
inline int parse_integer(const char* str) {
    std::string var(str ? str : "");
    try {
        return std::stoi(var);
    } catch (const std::exception&) {
        return INT32_MAX;
    }
}

inline std::string get_filename(const std::string& filePath) {
    auto index = filePath.find_last_of("/\\");
    if (std::string::npos == index) {
        return filePath;
    }
    return filePath.substr(index + 1);
}

inline int get_debug_level() {
    return parse_integer(std::getenv("OPENVINO_LOG_LEVEL"));
}
const int debug_level = get_debug_level();
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
    void set_prefix(std::string prefix);
    void set_suffix(std::string suffix);
    void set_log_level(LogLevel logLevel);

    template <typename... Args>
    void do_log(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
        const char* func, long line, const char* tag, const char* fmt, Args... args);
    void do_run(LogLevel level, const LogTask& task);
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
    std::string log_name;
    std::string log_path;
    std::string prefix;
    std::string suffix;
    uint32_t log_level;
    static uint32_t default_log_level;
    static std::vector<std::string> valid_format;
};

inline Log::Log()
    : log_level(default_log_level) {
    switch (debug_level) {
        case 0: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_NONE);
                    break;
                }
        case 1: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_FATAL);
                    break;
                }
        case 2: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_ERROR);
                    break;
                }
        case 3: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_WARNING);
                    break;
                }
        case 4: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_INFO);
                    break;
                }
        case 5: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_DEBUG);
                    break;
                }
        case 6: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_TRACE);
                    break;
                }
        case 7: {
                    log_level = static_cast<uint32_t>(LogLevel::LOG_FREQUENT);
                    break;
                }
        default:
                break;
    }
}

inline void Log::set_prefix(std::string prefix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    prefix = std::move(prefix_);
}

inline void Log::set_suffix(std::string suffix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    suffix = std::move(suffix_);
}

inline void Log::set_log_level(LogLevel loglevel_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    log_level = static_cast<uint32_t>(loglevel_);
}

inline void Log::print(std::stringstream& stream) {
    std::cout << stream.str() << std::endl << std::flush;
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
                        auto iter = std::find(valid_format.begin(), valid_format.end(), fmtStr);
                        if (iter != valid_format.end()) {
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
inline void Log::do_log(bool on, bool isTraceCallStack, LogLevel level, const char* levelStr, const char* file,
    const char* func, const long line, const char* tag, const char* fmt, Args... args) {

    if (!(static_cast<uint32_t>(level) & static_cast<uint32_t>(log_level)) || !on) {
        return;
    }

    std::stringstream stream;
    stream << colorBegin(level) << prefix << '[' << time_utils::get_current_time() << ']';

    if (level < LogLevel::ERROR) {
        stream << levelStr[0];
    } else {
        stream << levelStr;
    }
    stream << '[' << get_filename(file) << ':' << line << ']';

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

inline void Log::do_run(LogLevel level, const LogTask& task) {
    if (!(static_cast<uint32_t>(level) & static_cast<uint32_t>(log_level))) {
        return;
    }
    task();
}

inline std::string Log::colorBegin(auto_plugin::LogLevel loglevel) {
    if (loglevel == LogLevel::WARN) {
        return std::string(CYAN);
    }
    if (loglevel == LogLevel::ERROR || loglevel == LogLevel::FATAL) {
        return std::string(RED);
    }
    return std::string(DEFAULT_COLOR);
}

inline std::string Log::colorEnd(auto_plugin::LogLevel loglevel) {
    if (loglevel == LogLevel::WARN || loglevel == LogLevel::ERROR || loglevel == LogLevel::FATAL) {
        return std::string(COL_END);
    }
    return {};
}
} // namespace auto_plugin
} // namespace ov

#endif //AUTOPLUGIN_LOG_H
