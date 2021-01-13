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

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#ifdef _WIN32
#include <windows.h>
// windows.h must be before processthreadsapi.h so we need this comment
#include <processthreadsapi.h>
#define getpid() GetCurrentProcessId()
#else
#include <unistd.h>
#endif

#include <ngraph/ngraph_visibility.hpp>

namespace ngraph
{
    namespace event
    {
        class Duration;
        class Object;
        class Manager;
    }
}

//
// This class records timestamps for a given user defined event and
// produces output in the chrome tracing format that can be used to view
// the events of a running program
//
// Following is the format of a trace event
//
// {
//   "name": "myName",
//   "cat": "category,list",
//   "ph": "B",
//   "ts": 12345,
//   "pid": 123,
//   "tid": 456,
//   "args": {
//     "someArg": 1,
//     "anotherArg": {
//       "value": "my value"
//     }
//   }
// }
//
// The trace file format is defined here:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
//
// The trace file can be viewed by Chrome browser using the
// URL: chrome://tracing/
//
// More information about this is at:
// http://dev.chromium.org/developers/how-tos/trace-event-profiling-tool

class ngraph::event::Manager
{
    friend class Duration;
    friend class Object;

public:
    static void open(const std::string& path = "runtime_event_trace.json");
    static void close();
    static bool is_tracing_enabled() { return s_tracing_enabled; }
    static void enable_event_tracing();
    static void disable_event_tracing();
    static bool is_event_tracing_enabled();

private:
    static std::ofstream& get_output_stream();
    static const std::string& get_process_id();
    static size_t get_current_microseconds()
    {
        return std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000;
    }
    static std::string get_thread_id();
    static std::mutex& get_mutex() { return s_file_mutex; }
    static std::ostream s_ostream;
    static std::mutex s_file_mutex;
    static bool s_tracing_enabled;
};

class NGRAPH_API ngraph::event::Duration
{
public:
    explicit Duration(const std::string& name,
                      const std::string& category,
                      const std::string& args = "");
    ~Duration() { write(); }
    /// \brief stop the timer without writing the data to the log file. To write the data
    /// call the `write` method
    /// Calls to stop() are optional
    void stop();

    /// \brief write the log data to the log file for this event
    /// This funtion has an implicit stop() if stop() has not been previously called
    void write();

    Duration(const Duration&) = delete;
    Duration& operator=(Duration const&) = delete;

private:
    std::string to_json() const;
    size_t m_start{0};
    size_t m_stop{0};
    std::string m_name;
    std::string m_category;
    std::string m_args;
};

class ngraph::event::Object
{
public:
    Object(const std::string& name, const std::string& args);
    void snapshot(const std::string& args);
    void destroy();

private:
    void write_snapshot(std::ostream& out, const std::string& args);
    const std::string m_name;
    size_t m_id{0};
};
