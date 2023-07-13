// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/log.hpp"

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "ngraph/distributed.hpp"
#include "ngraph/env_util.hpp"
#include "openvino/util/file_util.hpp"

using namespace std;
using namespace ngraph;

void ngraph::default_logger_handler_func(const string& s) {
    cout << s + "\n";
}

LogHelper::LogHelper(LOG_TYPE type, const char* file, int line, function<void(const string&)> handler_func)
    : m_handler_func(handler_func) {
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
    }

    time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = gmtime(&tt);
    if (tm) {
        char buffer[256];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
        m_stream << buffer << " ";
    }

    m_stream << file;
    m_stream << " " << line;
    m_stream << "\t";
}

LogHelper::~LogHelper() {
#ifdef ENABLE_OPENVINO_DEBUG
    if (m_handler_func) {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
#endif
}

NGRAPH_SUPPRESS_DEPRECATED_START
const char* ngraph::trim_file_name(const char* const fname) {
    return ov::util::trim_file_name(fname);
}
NGRAPH_SUPPRESS_DEPRECATED_END
