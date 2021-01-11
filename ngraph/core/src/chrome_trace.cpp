//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "ngraph/chrome_trace.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

static bool read_tracing_env_var()
{
    static const bool is_enabled = getenv_bool("NGRAPH_ENABLE_TRACING");

    return is_enabled;
}

mutex event::Manager::s_file_mutex;
bool event::Manager::s_tracing_enabled = read_tracing_env_var();

event::Duration::Duration(const string& name, const string& category, const string& args)
{
    if (Manager::is_tracing_enabled())
    {
        m_start = Manager::get_current_microseconds();
        m_stop = 0;
        m_name = name;
        m_category = category;
        m_args = args;
    }
}

void event::Duration::stop()
{
    if (Manager::is_tracing_enabled())
    {
        m_stop = Manager::get_current_microseconds();
    }
}

void event::Duration::write()
{
    if (Manager::is_tracing_enabled())
    {
        size_t stop_time = (m_stop != 0 ? m_stop : Manager::get_current_microseconds());

        lock_guard<mutex> lock(Manager::get_mutex());

        ofstream& out = event::Manager::get_output_stream();
        string str;
        if (out.is_open() == false)
        {
            event::Manager::open();
        }
        else
        {
            str += ",\n";
        }

        str +=
            R"({"name":")" + m_name + R"(","cat":")" + m_category + R"(","ph":"X","pid":)" +
            Manager::get_process_id() + R"(,"tid":)" + Manager::get_thread_id() +
            R"(,"ts":)" + to_string(m_start) + R"(,"dur":)" + to_string(stop_time - m_start);
        if (!m_args.empty())
        {
            str += R"(,"args":)" + m_args;
        }
        str += "}";
        out << str;
    }
}

event::Object::Object(const string& name, const string& args)
    : m_name{name}
    , m_id{static_cast<size_t>(chrono::high_resolution_clock::now().time_since_epoch().count())}
{
    if (Manager::is_tracing_enabled())
    {
        lock_guard<mutex> lock(Manager::get_mutex());

        ofstream& out = event::Manager::get_output_stream();
        string str;
        if (out.is_open() == false)
        {
            event::Manager::open();
        }
        else
        {
            str += ",\n";
        }
        str += R"({"name":")" + m_name + R"(","ph":"N","id":")" + to_string(m_id) +
               R"(","ts":)" + to_string(Manager::get_current_microseconds()) +
               R"(,"pid":)" + Manager::get_process_id() + R"(,"tid":)" + Manager::get_thread_id();
        if (!args.empty())
        {
            str += R"(,"args":)" + args;
        }
        str += "}";

        write_snapshot(out, args);
    }
}

void event::Object::snapshot(const string& args)
{
    if (Manager::is_tracing_enabled())
    {
        lock_guard<mutex> lock(Manager::get_mutex());

        ofstream& out = event::Manager::get_output_stream();
        if (out.is_open() == false)
        {
            event::Manager::open();
        }
        else
        {
            Manager::get_output_stream() << ",\n";
        }
        write_snapshot(out, args);
    }
}

void event::Object::write_snapshot(ostream& out, const string& args)
{
    string str = R"({"name":")" + m_name + R"(","ph":"O","id":")" + to_string(m_id) +
                 R"(","ts":)" + to_string(Manager::get_current_microseconds()) +
                 R"(,"pid":)" + Manager::get_process_id() + R"(,"tid":)" + Manager::get_thread_id();
    if (!args.empty())
    {
        str += R"(,"args":)" + args;
    }
    str += "}";
    out << str;
}

void event::Object::destroy()
{
    if (Manager::is_tracing_enabled())
    {
        lock_guard<mutex> lock(Manager::get_mutex());

        ofstream& out = event::Manager::get_output_stream();
        if (out.is_open() == false)
        {
            event::Manager::open();
        }
        else
        {
            Manager::get_output_stream() << ",\n";
        }
        string str = R"({"name":")" + m_name + R"(","ph":"D","id":")" + to_string(m_id) +
                     R"(","ts":)" + to_string(Manager::get_current_microseconds()) +
                     R"(,"pid":)" + Manager::get_process_id() + R"(,"tid":)" +
                     Manager::get_thread_id() + "}";
    }
}

void event::Manager::open(const string& path)
{
    ofstream& out = get_output_stream();
    if (out.is_open() == false)
    {
        out.open(path, ios_base::trunc);
        out << "[\n";
    }
}

void event::Manager::close()
{
    ofstream& out = get_output_stream();
    if (out.is_open())
    {
        out << "\n]\n";
        out.close();
    }
}

ofstream& event::Manager::get_output_stream()
{
    static ofstream s_event_log;
    return s_event_log;
}

const string& event::Manager::get_process_id()
{
    static const string s_pid = to_string(getpid());
    return s_pid;
}

void event::Manager::enable_event_tracing()
{
    s_tracing_enabled = true;
}

void event::Manager::disable_event_tracing()
{
    s_tracing_enabled = false;
}

bool event::Manager::is_event_tracing_enabled()
{
    return s_tracing_enabled;
}

string event::Manager::get_thread_id()
{
    thread::id tid = this_thread::get_id();
    static map<thread::id, string> tid_map;
    auto it = tid_map.find(tid);
    string rc;
    if (it == tid_map.end())
    {
        stringstream ss;
        ss << "\"" << tid << "\"";
        rc = ss.str();
        tid_map.insert({tid, rc});
    }
    else
    {
        rc = it->second;
    }
    return rc;
}
