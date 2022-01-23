// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MULTIDEVICEPLUGIN_TIMEUTILS_H
#define MULTIDEVICEPLUGIN_TIMEUTILS_H

#include <chrono>
#include <ctime>
#include <string>

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
namespace TimeUtils {
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

    bool isExpired() const {
        auto now = std::chrono::steady_clock::now();
        return now > m_deadline;
    }

private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_deadline;
};

template <class TimePoint>
float getTimeDiffInMs(TimePoint tp1, TimePoint tp2) {
    std::chrono::duration<float, std::milli> duration = (tp1 > tp2) ? tp1 - tp2 : tp2 - tp1;
    return duration.count();
}

bool localtimeSafe(const time_t* time, struct tm* result);

std::string getCurrentTime();
std::string putTime(std::chrono::system_clock::time_point tp, const char* format);
} // namespace TimeUtils
} // namespace MultiDevicePlugin

#endif //MULTIDEVICEPLUGIN_TIMEUTILS_H
