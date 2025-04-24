// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

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
#include "openvino/runtime/properties.hpp"

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
    std::string var(str ? str : "-1");
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

using LogTask = std::function<void()>;

class Log : public Singleton<Log> {
public:
    void set_prefix(std::string prefix);
    void set_suffix(std::string suffix);
    void set_log_level(ov::log::Level logLevel);

    template <typename... Args>
    void do_log(bool on, bool isTraceCallStack, ov::log::Level level, const char* levelStr, const char* file,
        const char* func, long line, const char* tag, const char* fmt, Args... args);
    void do_run(ov::log::Level level, const LogTask& task);
#ifdef MULTIUNITTEST
    Log(std::string unittest):Log() {
    }
#endif

private:
    Log();
    friend Singleton<Log>;
    static std::string colorBegin(ov::log::Level logLevel);
    static std::string colorEnd(ov::log::Level logLevel);
    void checkFormat(const char* fmt);
    MOCKTESTMACRO void print(std::stringstream& stream);

private:
    std::mutex mutex;
    std::string log_name;
    std::string log_path;
    std::string prefix;
    std::string suffix;
    ov::log::Level log_level;
    static ov::log::Level default_log_level;
    static std::vector<std::string> valid_format;
};

inline Log::Log()
    : log_level(default_log_level) {
    log_level = ov::log::Level(debug_level);
}

inline void Log::set_prefix(std::string prefix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    prefix = std::move(prefix_);
}

inline void Log::set_suffix(std::string suffix_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    suffix = std::move(suffix_);
}

inline void Log::set_log_level(ov::log::Level loglevel_) {
    std::lock_guard<std::mutex> autoLock(mutex);
    log_level = loglevel_;
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
inline void Log::do_log(bool on, bool isTraceCallStack, ov::log::Level level, const char* levelStr, const char* file,
    const char* func, const long line, const char* tag, const char* fmt, Args... args) {

    if (level > log_level || !on) {
        return;
    }

    std::stringstream stream;
    stream << colorBegin(level) << prefix << '[' << time_utils::get_current_time() << ']';

    if (level > ov::log::Level::ERR) {
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

inline void Log::do_run(ov::log::Level level, const LogTask& task) {
    if (level > log_level)
        return;
    task();
}

inline std::string Log::colorBegin(ov::log::Level loglevel) {
    if (loglevel == ov::log::Level::WARNING) {
        return std::string(CYAN);
    }
    if (loglevel == ov::log::Level::ERR) {
        return std::string(RED);
    }
    return std::string(DEFAULT_COLOR);
}

inline std::string Log::colorEnd(ov::log::Level loglevel) {
    if (loglevel == ov::log::Level::WARNING || loglevel == ov::log::Level::ERR) {
        return std::string(COL_END);
    }
    return {};
}
} // namespace auto_plugin
} // namespace ov