// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <vector>

namespace ov {
namespace util {

enum class LOG_TYPE {
    _LOG_TYPE_ERROR,
    _LOG_TYPE_WARNING,
    _LOG_TYPE_INFO,
    _LOG_TYPE_DEBUG,
};

class LogHelper {
public:
    LogHelper(LOG_TYPE, const char* file, int line, std::function<void(const std::string&)> m_handler_func);
    ~LogHelper();

    std::ostream& stream() {
        return m_stream;
    }

private:
    std::function<void(const std::string&)> m_handler_func;
    std::stringstream m_stream;
};

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
#endif

}  // namespace util
}  // namespace ov
