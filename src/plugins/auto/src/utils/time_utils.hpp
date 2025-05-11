// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <chrono>
#include <ctime>
#include <string>

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#else
#define MOCKTESTMACRO
#endif

namespace ov {
namespace auto_plugin {
namespace time_utils {
class StopWatch {
public:
    explicit StopWatch(int seconds) {
        m_start = std::chrono::steady_clock::now();
        m_deadline = m_start + std::chrono::seconds(seconds);
    }

    void reset(int seconds) {
        m_start = std::chrono::steady_clock::now();
        m_deadline = m_start + std::chrono::seconds(seconds);
    }

    bool is_expired() const {
        auto now = std::chrono::steady_clock::now();
        return now > m_deadline;
    }

private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_deadline;
};

template <class TimePoint>
float get_time_diff_in_ms(TimePoint tp1, TimePoint tp2) {
    std::chrono::duration<float, std::milli> duration = (tp1 > tp2) ? tp1 - tp2 : tp2 - tp1;
    return duration.count();
}

bool local_time_safe(const time_t* time, struct tm* result);

std::string get_current_time();
std::string put_time(std::chrono::system_clock::time_point tp, const char* format);
} // namespace time_utils
} // namespace auto_plugin
} // namespace ov
