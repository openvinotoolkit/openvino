//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "ngraph/distributed.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

void ngraph::default_logger_handler_func(const string& s)
{
    cout << s + "\n";
}

LogHelper::LogHelper(LOG_TYPE type,
                     const char* file,
                     int line,
                     function<void(const string&)> handler_func)
    : m_handler_func(handler_func)
{
    switch (type)
    {
    case LOG_TYPE::_LOG_TYPE_ERROR: m_stream << "[ERR] "; break;
    case LOG_TYPE::_LOG_TYPE_WARNING: m_stream << "[WARN] "; break;
    case LOG_TYPE::_LOG_TYPE_INFO: m_stream << "[INFO] "; break;
    case LOG_TYPE::_LOG_TYPE_DEBUG: m_stream << "[DEBUG] "; break;
    }

    time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = gmtime(&tt);
    if (tm)
    {
        char buffer[256];
        strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
        m_stream << buffer << " ";
    }

    m_stream << file;
    m_stream << " " << line;
    m_stream << "\t";
}

LogHelper::~LogHelper()
{
    if (m_handler_func)
    {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
}
