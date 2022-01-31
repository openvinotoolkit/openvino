// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <string>
#include <cstring>
#include <functional>
#include <stdexcept>

#define MVLOG_UNIT_NAME watchdog
#include "XLinkLog.h"

#if defined(_WIN32)
#include "win_synchapi.h"
#endif // defined(_WIN32)

namespace Watchdog {

/**
 * @brief represents watchdog device interface to be registered within watchdog worker
 */
class IDevice {
public:
    using time_point = std::chrono::steady_clock::time_point;

    virtual ~IDevice() = default;

    /**
     * @brief watchdog request device to keep alive with current timestamp
     */
    virtual void keepAlive(const time_point &current_time) noexcept = 0;
    /**
     * @brief means we need to ping it after corresponding time
     */
    virtual std::chrono::milliseconds dueIn(const time_point &current_time) const noexcept = 0;
    /**
     * @brief whether device is hanging
     */
    virtual bool isTimeout() const noexcept = 0;
    /**
     * @brief gets opaque handle that clearly identifies watchdog device, ex.: usb connection identifier
     */
    virtual void *getHandle() const noexcept = 0;
};

class AutoScope {
public:
    explicit AutoScope(const std::function<void()>& func) : _func(func) {}
    ~AutoScope() { _func(); }

    AutoScope(const AutoScope&) = delete;
    AutoScope(AutoScope&&) = delete;
    AutoScope& operator=(const AutoScope&) = delete;
    AutoScope& operator=(AutoScope&&) = delete;
private:
    std::function<void()> _func;
};

class CustomUniqueLock {
public:
    explicit CustomUniqueLock(pthread_mutex_t* mutex)
        :m_mutex(mutex) {
        if(m_mutex == nullptr) {
            throw std::runtime_error("mutex should not be null");
        }

        int rc = pthread_mutex_lock(m_mutex);
        if (rc != 0) {
            throw std::runtime_error(std::string("failed to lock mutex. rc: ") + strerror(rc));
        }
    };

    ~CustomUniqueLock() {
        int rc = pthread_mutex_unlock(m_mutex);
        if (rc != 0) {
            mvLog(MVLOG_ERROR, "failed to unlock mutex. rc: %s", strerror(rc));
        }
    }

    CustomUniqueLock(const CustomUniqueLock&) = delete;
    CustomUniqueLock(const CustomUniqueLock&&) = delete;
    CustomUniqueLock& operator=(const CustomUniqueLock&) = delete;
    CustomUniqueLock& operator=(const CustomUniqueLock&&) = delete;

private:
    pthread_mutex_t* m_mutex = nullptr;
};

}  // namespace Watchdog
