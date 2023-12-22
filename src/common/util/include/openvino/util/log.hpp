// Copyright (C) 2018-2023 Intel Corporation
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

#define OPENVINO_ERR                                               \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_ERROR,   \
                          __FILE__,                                \
                          __LINE__,                                \
                          ::ov::util::default_logger_handler_func) \
        .stream()

#define OPENVINO_WARN                                              \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_WARNING, \
                          __FILE__,                                \
                          __LINE__,                                \
                          ::ov::util::default_logger_handler_func) \
        .stream()

#define OPENVINO_INFO                                              \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_INFO,    \
                          __FILE__,                                \
                          __LINE__,                                \
                          ::ov::util::default_logger_handler_func) \
        .stream()

#define OPENVINO_DEBUG                                             \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_DEBUG,   \
                          __FILE__,                                \
                          __LINE__,                                \
                          ::ov::util::default_logger_handler_func) \
        .stream()
}  // namespace util
}  // namespace ov
