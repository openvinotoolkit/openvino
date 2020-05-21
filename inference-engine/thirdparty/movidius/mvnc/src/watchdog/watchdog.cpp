// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "watchdog.h"
#include "watchdogPrivate.hpp"

#include <thread>
#include <vector>
#include <ostream>
#include <iostream>
#include <atomic>
#include <memory>
#include <algorithm>
#include <unordered_map>

#define MVLOG_UNIT_NAME watchdog
#include "XLinkLog.h"

namespace {

using namespace std;
using namespace chrono;
using namespace Watchdog;

/**
 * @brief when device just added into watchdog, it should not be due interval at all
 */
class NoDueOnFirstCall : public IDevice {
public:
    NoDueOnFirstCall(IDevice* original) : m_originalPtr(original) {}

    void keepAlive(const time_point& current_time) noexcept override  {
        m_originalPtr->keepAlive(current_time);
        m_firstCall = true;
    }

    milliseconds dueIn(const time_point& current_time) const noexcept override {
        if (!m_firstCall) {
            return milliseconds::zero();
        }

        return m_originalPtr->dueIn(current_time);
    }

    bool isTimeout() const noexcept override {
        return m_originalPtr->isTimeout();
    }

    void* getHandle() const noexcept override {
        return m_originalPtr->getHandle();
    }

private:
    IDevice* m_originalPtr;
    bool m_firstCall = false;
};

class WatchdogImpl {
public:
    WatchdogImpl();
    ~WatchdogImpl();

    bool registerDevice(IDevice* device);
    bool removeDevice(IDevice* device);

    WatchdogImpl(const WatchdogImpl&) = delete;
    WatchdogImpl(WatchdogImpl&&) = delete;
    WatchdogImpl& operator = (const WatchdogImpl&) = delete;
    WatchdogImpl& operator = (WatchdogImpl&&) = delete;

private:
    void waitFor(const milliseconds sleepInterval);
    void watchdogRoutine() noexcept;

private:
    using Devices = std::vector<std::shared_ptr<IDevice>>;
    using DevicesMap = std::unordered_map<void*, std::shared_ptr<IDevice>>;

    Devices watchedDevices;
    DevicesMap removedDevices;
    std::atomic_bool threadRunning {false};

    pthread_mutex_t routineLock;
    pthread_cond_t  wakeUpPingThread;
    std::thread poolThread;
};

//------------- Watchdog implementation -------------

WatchdogImpl::WatchdogImpl() {
    int rc = pthread_mutex_init(&routineLock, NULL);
    if (rc != 0) {
        throw std::runtime_error("failed to initialize \"routineLock\" mutex. rc: " + std::to_string(rc));
    }

#if !(defined(__APPLE__) || defined(_WIN32))
    pthread_condattr_t attr;
    rc = pthread_condattr_init(&attr);
    if (rc != 0) {
        throw std::runtime_error("failed to initialize condition variable attribute. rc: " + std::to_string(rc));
    }

    AutoScope attrDestroy([&attr]{
        if (pthread_condattr_destroy(&attr) != 0)
            mvLog(MVLOG_ERROR, "Failed to destroy condition variable attribute.");
    });

    rc = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
    if (rc != 0) {
        throw std::runtime_error("failed to set condition variable clock. rc: " + std::to_string(rc));
    }
#endif // !(defined(__APPLE__) || defined(_WIN32))

    rc = pthread_cond_init(&wakeUpPingThread, NULL);
    if (rc != 0) {
        throw std::runtime_error("failed to initialize \"wakeUpPingThread\" condition variable. rc: " + std::to_string(rc));
    }
}

WatchdogImpl::~WatchdogImpl() {
    mvLog(MVLOG_INFO, "watchdog terminated\n");
    try
    {
        CustomUniqueLock lock {&routineLock};
        for (auto &item : watchedDevices) {
            mvLog(MVLOG_WARN, "[%p] device, stop watching due to watchdog termination\n", item->getHandle());
        }
    } catch (const std::exception & ex) {
        mvLog(MVLOG_ERROR, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "unknown error");
    }

    threadRunning = false;
    int rc = pthread_cond_broadcast(&wakeUpPingThread);
    if (rc != 0) {
        mvLog(MVLOG_WARN, "failed to unblock threads blocked on the \"wakeUpPingThread\". rc=%d", rc);
    }

    if (poolThread.joinable()) {
        poolThread.join();
    }

    rc = pthread_mutex_destroy(&routineLock);
    if (rc != 0) {
        mvLog(MVLOG_WARN, "failed to destroy the \"routineLock\". rc=%d", rc);
    }

    rc = pthread_cond_destroy(&wakeUpPingThread);
    if (rc != 0) {
        mvLog(MVLOG_WARN, "failed to destroy the \"wakeUpPingThread\". rc=%d", rc);
    }
}

bool WatchdogImpl::registerDevice(IDevice* device) {
    mvLog(MVLOG_INFO, "register device: %p\n", &device);

    CustomUniqueLock lock {&routineLock};

    if (!threadRunning) {
        if (poolThread.joinable()) {
            poolThread.join();
        }
        threadRunning = true;

        poolThread = std::thread([this]() {
            if (pthread_setname_np(
#ifndef __APPLE__
            pthread_self(),
#endif
            "WatchdogThread") != 0) {
                perror("Setting name for watchdog thread failed");
            }
            watchdogRoutine();
        });
    }

    auto it = std::find_if(std::begin(watchedDevices),
                           std::end(watchedDevices),
                           [&device](const std::shared_ptr<IDevice>& item) {
                               return item->getHandle() == device->getHandle();
                           });

    bool found = it != std::end(watchedDevices);
    if (!found) {
        watchedDevices.emplace_back(std::make_shared<NoDueOnFirstCall>(device));
    }

    int rc = pthread_cond_broadcast(&wakeUpPingThread);
    if (rc != 0) {
        mvLog(MVLOG_WARN, "failed to unblock threads blocked on the \"wakeUpPingThread\". rc=%d", rc);
    }

    return !found;
}

bool WatchdogImpl::removeDevice(IDevice* device) {
    mvLog(MVLOG_INFO, "remove device: %p\n", &device);

    CustomUniqueLock lock {&routineLock};

    auto it = std::find_if(std::begin(watchedDevices),
                           std::end(watchedDevices),
                           [&device](const std::shared_ptr<IDevice>& item) {
                               return item->getHandle() == device->getHandle();
                           });

    bool removed = it != std::end(watchedDevices);
    if (removed) {
        watchedDevices.erase(it);
    } else if (removedDevices.count(device->getHandle())) {
        removedDevices.erase(device->getHandle());
        removed = true;
    }

    // wake up thread since we might select removed device as nex to be ping, and there is no more devices available
    int rc = pthread_cond_broadcast(&wakeUpPingThread);
    if (rc != 0) {
        mvLog(MVLOG_WARN, "failed to unblock threads blocked on the \"wakeUpPingThread\". rc=%d", rc);
    }

    return removed;
}

void WatchdogImpl::waitFor(const milliseconds sleepInterval) {
    struct timespec timeToWait = {0, 0};

    const auto sec = std::chrono::duration_cast<std::chrono::seconds>(sleepInterval);

#if (defined(__APPLE__) || defined(_WIN32))
    timeToWait.tv_sec = sec.count();
    timeToWait.tv_nsec =
        std::chrono::duration_cast<std::chrono::nanoseconds>(sleepInterval).count() -
        std::chrono::nanoseconds(sec).count();
#else
    clock_gettime(CLOCK_MONOTONIC, &timeToWait);
    const auto secondInNanoSeconds = 1000000000L;
    const auto nsecSum = std::chrono::duration_cast<std::chrono::nanoseconds>(sleepInterval).count() -
                         std::chrono::nanoseconds(sec).count() + timeToWait.tv_nsec;
    timeToWait.tv_sec += sec.count() + nsecSum / secondInNanoSeconds;
    timeToWait.tv_nsec = nsecSum % secondInNanoSeconds;
#endif // (defined(__APPLE__) || defined(_WIN32))

#if defined(__APPLE__)
    const auto rc = pthread_cond_timedwait_relative_np(&wakeUpPingThread, &routineLock, &timeToWait);
#else
    const auto rc = pthread_cond_timedwait(&wakeUpPingThread, &routineLock, &timeToWait);
#endif // defined(__APPLE__)

    if (rc != 0 && rc != ETIMEDOUT) {
        throw std::runtime_error("Failed to perform wait in a loop for " + std::to_string(sleepInterval.count()) + " ms. rc: " + std::to_string(rc));
    }
}

void WatchdogImpl::watchdogRoutine() noexcept {
    try {
        mvLog(MVLOG_INFO, "thread started\n");

        milliseconds sleepInterval;
        CustomUniqueLock lock{&routineLock};

        do {
            for (auto deviceIt = watchedDevices.begin(); deviceIt != watchedDevices.end();) {
                auto &device = *deviceIt;
                auto isReady = device->dueIn(steady_clock::now()).count() <= 0;
                if (isReady) {
                    auto now = steady_clock::now();
                    device->keepAlive(steady_clock::now());
                    mvLog(MVLOG_DEBUG, "ping completed in %ld ms\n",
                          duration_cast<std::chrono::milliseconds>(steady_clock::now() - now).count());
                }
                if (device->isTimeout()) {
                    mvLog(MVLOG_ERROR, "[%p] device, not respond, removing from watchdog\n", device->getHandle());
                    // marking device as deleted, to prevent double resource free from wd_unregister_device
                    removedDevices[device->getHandle()] = device;
                    deviceIt = watchedDevices.erase(deviceIt);
                } else {
                    ++deviceIt;
                }
            }
            auto currentTime = steady_clock::now();
            auto minInterval = std::min_element(watchedDevices.begin(), watchedDevices.end(),
                                                [&currentTime](const Devices::value_type& device1,
                                                               const Devices::value_type& device2) {
                                                    return device1->dueIn(currentTime).count() <
                                                            device2->dueIn(currentTime).count();
                                                });
            // if for some reason we have empty devices list but watchdog is active
            if (minInterval == watchedDevices.end()) {
                mvLog(MVLOG_INFO, "no active devices to watch, stopping  Watchdog thread\n");
                threadRunning = false;
                break;
            }

            sleepInterval = (*minInterval)->dueIn(currentTime);
            if (sleepInterval.count() <= 0) {
                continue;
            }

            mvLog(MVLOG_DEBUG, "sleep interval = %ld ms\n", sleepInterval.count());

            waitFor(sleepInterval);

            mvLog(MVLOG_DEBUG, "waiting completed in  %ld ms\n",
                  duration_cast<std::chrono::milliseconds>(steady_clock::now() - currentTime).count());

        } while (threadRunning);
    } catch (const std::exception &ex) {
        mvLog(MVLOG_ERROR, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "unknown error");
    }

    mvLog(MVLOG_INFO, "thread ended\n");
}

}  // namespace

struct _WatchdogHndl_t {
    WatchdogImpl* m_watchdog;
};

wd_error_t watchdog_create(WatchdogHndl_t** out_watchdogHndl) {
    if (out_watchdogHndl == nullptr) {
        return WD_NOTINITIALIZED;
    }

    *out_watchdogHndl = nullptr;
    auto tmpWdHndl =
        static_cast<WatchdogHndl_t*>(malloc(sizeof(WatchdogHndl_t)));
    if(tmpWdHndl == nullptr) {
        return WD_FAIL;
    }

    try {
        tmpWdHndl->m_watchdog = new WatchdogImpl();
        *out_watchdogHndl = tmpWdHndl;
        return WD_ERRNO;
    } catch (const std::exception& ex) {
        mvLog(MVLOG_ERROR, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "unknown error");
    }

    free(tmpWdHndl);
    return WD_FAIL;
}

void watchdog_destroy(WatchdogHndl_t* watchdogHndl) {
    if (watchdogHndl == nullptr) {
        return;
    }

    if (watchdogHndl->m_watchdog != nullptr) {
        delete(watchdogHndl->m_watchdog);
    }

    free(watchdogHndl);
}

wd_error_t watchdog_register_device(WatchdogHndl_t* watchdogHndl, WdDeviceHndl_t* deviceHandle) {
    if (watchdogHndl == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog handle is null\n");
        return WD_NOTINITIALIZED;
    }

    if (deviceHandle == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog device handle is null\n");
        return WD_NOTINITIALIZED;
    }

    if (deviceHandle->m_device == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog device not initialized. handle=%p\n", deviceHandle);
        return WD_NOTINITIALIZED;
    }

    try {
        WatchdogImpl* watchdog = watchdogHndl->m_watchdog;
        auto device = reinterpret_cast<IDevice*>(deviceHandle->m_device);
        if (!watchdog->registerDevice(device)) {
            mvLog(MVLOG_WARN, "cannot register device\n");
            return WD_FAIL;
        }
        return WD_ERRNO;
    } catch (const std::exception & ex) {
        mvLog(MVLOG_ERROR, "failed to register device: %s\n", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "failed to register device (%p)\n", deviceHandle);
    }

    return WD_FAIL;
}

wd_error_t watchdog_unregister_device(WatchdogHndl_t* watchdogHndl, WdDeviceHndl_t* deviceHandle) {
    if (watchdogHndl == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog handle is null\n");
        return WD_NOTINITIALIZED;
    }

    if (deviceHandle == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog device handle is null\n");
        return WD_NOTINITIALIZED;
    }

    if (deviceHandle->m_device == nullptr) {
        mvLog(MVLOG_ERROR, "watchdog device not initialized. handle=%p\n", deviceHandle);
        return WD_NOTINITIALIZED;
    }

    try {
        WatchdogImpl* watchdog = watchdogHndl->m_watchdog;
        auto device = reinterpret_cast<IDevice*>(deviceHandle->m_device);
        if (!watchdog->removeDevice(device)) {
            mvLog(MVLOG_WARN, "cannot remove device\n");
            return WD_FAIL;
        }
        return WD_ERRNO;
    } catch (const std::exception & ex) {
        mvLog(MVLOG_ERROR, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "unknown error");
    }

    return WD_FAIL;
}
