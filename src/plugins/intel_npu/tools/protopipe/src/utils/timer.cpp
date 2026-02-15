//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "timer.hpp"
#include "utils.hpp"

#include <thread>

#if defined(_WIN32)
#include <windows.h>

class WinTimer : public SleepTimer {
public:
    WinTimer(bool disable_high_resolution_timer);
    void wait(std::chrono::microseconds time) override;
    ~WinTimer();

private:
    HANDLE m_handle = nullptr;
};

WinTimer::WinTimer(bool disable_high_resolution_timer) {
    // FIXME: It should be called once.
    timeBeginPeriod(1);
    m_handle = CreateWaitableTimerEx(
            NULL, NULL, disable_high_resolution_timer ? 0 : CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
}

void WinTimer::wait(std::chrono::microseconds time) {
    LARGE_INTEGER li;
    using ns_t = std::chrono::nanoseconds;
    using ns_100_t = std::chrono::duration<ns_t::rep, std::ratio_multiply<std::ratio<100>, ns_t::period>>;

    li.QuadPart = -std::chrono::duration_cast<ns_100_t>(time).count();
    if (!SetWaitableTimer(m_handle, &li, 0, NULL, NULL, false)) {
        CloseHandle(m_handle);
        throw std::logic_error("WinTimer failed to setup");
    }

    if (WaitForSingleObject(m_handle, INFINITE) != WAIT_OBJECT_0) {
        CloseHandle(m_handle);
        throw std::logic_error("WinTimer failed to sleep");
    }
}

WinTimer::~WinTimer() {
    CancelWaitableTimer(m_handle);
    CloseHandle(m_handle);
}

#endif  // defined(_WIN32)

class ChronoTimer : public SleepTimer {
    void wait(std::chrono::microseconds time) override;
};

void ChronoTimer::wait(std::chrono::microseconds time) {
    std::this_thread::sleep_for(time);
}

SleepTimer::Ptr SleepTimer::create(bool disable_high_resolution_timer) {
#if defined(_WIN32)
    return std::make_shared<WinTimer>(disable_high_resolution_timer);
#else
    return std::make_shared<ChronoTimer>();
#endif
}

void BusyTimer::wait(std::chrono::microseconds time) {
    utils::busyWait(time);
}
