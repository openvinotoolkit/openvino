// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <chrono>
#include <cstdarg>
#include <deque>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#if defined(__linux) || defined(__APPLE__)
#    include <sys/time.h>
#    include <unistd.h>
#endif
#include <ngraph/ngraph_visibility.hpp>
#include <vector>

#include "ngraph/deprecated.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
class NGRAPH_API_DEPRECATED ConstString {
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

NGRAPH_API_DEPRECATED
constexpr const char* find_last(ConstString s, size_t offset, char ch) {
    return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1) : find_last(s, offset - 1, ch));
}

NGRAPH_API_DEPRECATED
constexpr const char* find_last(ConstString s, char ch) {
    return find_last(s, s.size() - 1, ch);
}

NGRAPH_API_DEPRECATED
constexpr const char* get_file_name(ConstString s) {
    return find_last(s, '/');
}
NGRAPH_API_DEPRECATED
NGRAPH_API
const char* trim_file_name(const char* const fname);

enum class LOG_TYPE {
    _LOG_TYPE_ERROR,
    _LOG_TYPE_WARNING,
    _LOG_TYPE_INFO,
    _LOG_TYPE_DEBUG,
};

class NGRAPH_API_DEPRECATED NGRAPH_API LogHelper {
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

class NGRAPH_API_DEPRECATED Logger {
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

NGRAPH_API_DEPRECATED
NGRAPH_API
void default_logger_handler_func(const std::string& s);

NGRAPH_SUPPRESS_DEPRECATED_END

#define NGRAPH_ERR                                         \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_ERROR,   \
                      ngraph::trim_file_name(__FILE__),    \
                      __LINE__,                            \
                      ngraph::default_logger_handler_func) \
        .stream()

#define NGRAPH_WARN                                        \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_WARNING, \
                      ngraph::trim_file_name(__FILE__),    \
                      __LINE__,                            \
                      ngraph::default_logger_handler_func) \
        .stream()

#define NGRAPH_INFO                                        \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_INFO,    \
                      ngraph::trim_file_name(__FILE__),    \
                      __LINE__,                            \
                      ngraph::default_logger_handler_func) \
        .stream()

#define NGRAPH_DEBUG                                       \
    ngraph::LogHelper(ngraph::LOG_TYPE::_LOG_TYPE_DEBUG,   \
                      ngraph::trim_file_name(__FILE__),    \
                      __LINE__,                            \
                      ngraph::default_logger_handler_func) \
        .stream()
}  // namespace ngraph
