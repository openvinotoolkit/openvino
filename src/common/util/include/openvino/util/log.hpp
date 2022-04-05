// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <vector>

namespace ov {
namespace util {
class ConstString {
public:
    template <size_t SIZE>
    constexpr ConstString(const char (&p)[SIZE]) : m_string(p),
                                                   m_size(SIZE) {}

    constexpr char operator[](size_t i) const {
        return i < m_size ? m_string[i] : throw std::out_of_range("");
    }
    constexpr const char* get_ptr(size_t offset) const {
        return offset < m_size ? &m_string[offset] : m_string;
    }
    constexpr size_t size() const {
        return m_size;
    }

private:
    const char* m_string;
    size_t m_size;
};

constexpr const char* find_last(ConstString s, size_t offset, char ch) {
    return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1) : find_last(s, offset - 1, ch));
}

constexpr const char* find_last(ConstString s, char ch) {
    return find_last(s, s.size() - 1, ch);
}

constexpr const char* get_file_name(ConstString s) {
    return find_last(s, '/');
}
constexpr const char* trim_file_name(ConstString root, ConstString s) {
    return s.get_ptr(root.size());
}
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

#ifndef PROJECT_ROOT_DIR
#    define PROJECT_ROOT_DIR ""
#endif

#define OPENVINO_ERR                                                              \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_ERROR,                  \
                          ::ov::util::trim_file_name(PROJECT_ROOT_DIR, __FILE__), \
                          __LINE__,                                               \
                          ::ov::util::default_logger_handler_func)                \
        .stream()

#define OPENVINO_WARN                                                             \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_WARNING,                \
                          ::ov::util::trim_file_name(PROJECT_ROOT_DIR, __FILE__), \
                          __LINE__,                                               \
                          ::ov::util::default_logger_handler_func)                \
        .stream()

#define OPENVINO_INFO                                                             \
    ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_INFO,                   \
                          ::ov::util::trim_file_name(PROJECT_ROOT_DIR, __FILE__), \
                          __LINE__,                                               \
                          ::ov::util::default_logger_handler_func)                \
        .stream()

#ifdef ENABLE_OPENVINO_DEBUG
#    define OPENVINO_DEBUG                                                            \
        ::ov::util::LogHelper(::ov::util::LOG_TYPE::_LOG_TYPE_DEBUG,                  \
                              ::ov::util::trim_file_name(PROJECT_ROOT_DIR, __FILE__), \
                              __LINE__,                                               \
                              ::ov::util::default_logger_handler_func)                \
            .stream()
#else

struct NullLogger {};

template <typename T>
NullLogger&& operator<<(NullLogger&& logger, T&&) {
    return std::move(logger);
}

template <typename T>
NullLogger&& operator<<(NullLogger&& logger, const T&) {
    return std::move(logger);
}

inline NullLogger&& operator<<(
    NullLogger&& logger,
    std::basic_ostream<char, std::char_traits<char>>& (&)(std::basic_ostream<char, std::char_traits<char>>&)) {
    return std::move(logger);
}

#    define OPENVINO_DEBUG \
        ::ov::util::NullLogger {}
#endif
}  // namespace util
}  // namespace ov
