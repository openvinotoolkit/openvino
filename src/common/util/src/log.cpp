// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log.hpp"

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <mutex>

#include "openvino/util/file_util.hpp"

void ov::util::default_logger_handler_func(const std::string& s) {
    std::cout << s << std::endl;
}

ov::util::LogHelper::LogHelper(LOG_TYPE type,
                               const char* file,
                               int line,
                               std::function<void(const std::string&)> handler_func)
    : m_handler_func(std::move(handler_func)) {
    switch (type) {
    case LOG_TYPE::_LOG_TYPE_ERROR:
        m_stream << "[ERR] ";
        break;
    case LOG_TYPE::_LOG_TYPE_WARNING:
        m_stream << "[WARN] ";
        break;
    case LOG_TYPE::_LOG_TYPE_INFO:
        m_stream << "[INFO] ";
        break;
    case LOG_TYPE::_LOG_TYPE_DEBUG:
        m_stream << "[DEBUG] ";
        break;
    case LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY:
        break;
    }

    if (type != LOG_TYPE::_LOG_TYPE_DEBUG_EMPTY) {
        {
            static std::mutex m;
            time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::lock_guard<std::mutex> lock(m);
            auto tm = gmtime(&tt);
            if (tm) {
                char buffer[256];
                strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
                m_stream << buffer << " ";
            }
        }

        m_stream << util::trim_file_name(file);
        m_stream << " " << line;
        m_stream << "\t";
    }
}

ov::util::LogHelper::~LogHelper() {
#ifdef ENABLE_OPENVINO_DEBUG
    if (m_handler_func) {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
#endif
}
