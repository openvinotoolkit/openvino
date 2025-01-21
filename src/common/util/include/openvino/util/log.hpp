// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <vector>
#include <unordered_set>

namespace ov {
namespace util {

enum class LOG_TYPE {
    _LOG_TYPE_ERROR,
    _LOG_TYPE_WARNING,
    _LOG_TYPE_INFO,
    _LOG_TYPE_DEBUG,
    _LOG_TYPE_DEBUG_EMPTY,
};

class LogHelper {
public:
    LogHelper(LOG_TYPE, const char* file, int line, std::function<void(const std::string&)> m_handler_func);
    ~LogHelper();

    std::ostream& stream() {
        return m_stream;
    }

// TODO: rework this in a separate custom string class with
// operators ++ and -- implemented to avoid unnecessary
// recreation of the level string
static std::string level_string_fun(int level) {
    std::string res = "│  ";
    res.reserve(3 + 3 * level);
    for (int i = 0; i < level; ++i) {
        res += "│  ";
    }
    return res;
}

inline const std::unordered_set<std::string> get_matchers_to_log() { return m_matchers_to_log; }

private:
    std::function<void(const std::string&)> m_handler_func;
    std::stringstream m_stream;
// If you want to log only specific matchers, put their names in this list and recompile.
// The empty list would mean logging of all matchers.
    std::unordered_set<std::string> m_matchers_to_log = {};
};

// // TODO: maybe move to a different place
#define level_string(level) \
    ov::util::LogHelper::level_string_fun(level)

class Logger {
    friend class LogHelper;

public:
    static void set_log_path(const std::string& path);
    static void start();
    static void stop();

private:
    static void log_item(const std::string& s);
    static void process_event(const std::string& s);
    static void thread_entry(void* param);
    static std::string m_log_path;
    static std::deque<std::string> m_queue;
};

void default_logger_handler_func(const std::string& s);
void default_logger_handler_func_length(const std::string& s);

#ifdef ENABLE_OPENVINO_DEBUG
/* Template function _write_all_to_stream has duplicates
 * It's defined in:
 * intel_cpu/src/utils/debug_capabilities and src/core/include/openvino/core/except.hpp
 * To prevent loop dependencies this code is currently duplicated, but should be moved
 * from intel_cpu, core and this place into one common library.
 */
static inline std::ostream& _write_all_to_stream(std::ostream& os) {
    return os;
}

template <typename T, typename... TS>
static inline std::ostream& _write_all_to_stream(std::ostream& os, const T& arg, TS&&... args) {
    return ov::util::_write_all_to_stream(os << arg, std::forward<TS>(args)...);
}

#    define OPENVINO_LOG_STREAM(OPENVINO_HELPER_LOG_TYPE)                     \
        ::ov::util::LogHelper(::ov::util::LOG_TYPE::OPENVINO_HELPER_LOG_TYPE, \
                              __FILE__,                                       \
                              __LINE__,                                       \
                              ::ov::util::default_logger_handler_func)        \
            .stream()

// Same as above, but no call to .stream() and no newline if the stream is empty
#    define OPENVINO_LOG(OPENVINO_HELPER_LOG_TYPE)                                \
        ::ov::util::LogHelper(::ov::util::LOG_TYPE::OPENVINO_HELPER_LOG_TYPE,     \
                              __FILE__,                                           \
                              __LINE__,                                           \
                              ::ov::util::default_logger_handler_func_length)     \

#    define OPENVINO_ERR(...)                                                                  \
        do {                                                                                   \
            ov::util::_write_all_to_stream(OPENVINO_LOG_STREAM(_LOG_TYPE_ERROR), __VA_ARGS__); \
        } while (0)

#    define OPENVINO_WARN(...)                                                                   \
        do {                                                                                     \
            ov::util::_write_all_to_stream(OPENVINO_LOG_STREAM(_LOG_TYPE_WARNING), __VA_ARGS__); \
        } while (0)

#    define OPENVINO_INFO(...)                                                                \
        do {                                                                                  \
            ov::util::_write_all_to_stream(OPENVINO_LOG_STREAM(_LOG_TYPE_INFO), __VA_ARGS__); \
        } while (0)

#    define OPENVINO_DEBUG(...)                                                                \
        do {                                                                                   \
            ov::util::_write_all_to_stream(OPENVINO_LOG_STREAM(_LOG_TYPE_DEBUG), __VA_ARGS__); \
        } while (0)

#    define OPENVINO_DEBUG_EMPTY(matcher_ptr, ...)                                                                     \
        do {                                                                                                           \
            ov::util::LogHelper logger = OPENVINO_LOG(_LOG_TYPE_DEBUG_EMPTY);                                          \
            if (logger.get_matchers_to_log().empty()) {                                                                \
                ov::util::_write_all_to_stream(logger.stream(), __VA_ARGS__);                                          \
            } else {                                                                                                   \
                if (logger.get_matchers_to_log().find(matcher_ptr->get_name()) != logger.get_matchers_to_log().end()) {\
                    ov::util::_write_all_to_stream(logger.stream(), __VA_ARGS__);                                      \
                }                                                                                                      \
            }                                                                                                          \
        } while (0)
#else
#    define OPENVINO_ERR(...) \
        do {                  \
        } while (0)
#    define OPENVINO_WARN(...) \
        do {                   \
        } while (0)
#    define OPENVINO_INFO(...) \
        do {                   \
        } while (0)
#    define OPENVINO_DEBUG(...) \
        do {                    \
        } while (0)
#    define OPENVINO_DEBUG_EMPTY(matcher_ptr, ...) \
        do {                                       \
        } while (0)
#endif

}  // namespace util
}  // namespace ov
