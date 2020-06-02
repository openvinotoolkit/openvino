// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>
#include <future>
#include <vector>
#include <ostream>
#include <iostream>
#include <utility>
#include <watchdog.h>
#include <watchdogPrivate.hpp>
#include <algorithm>
#include <memory>
#include <string>
#include <cstring>
#include <ncCommPrivate.h>
#include <mvnc.h>
#include <ncPrivateTypes.h>
#include <list>

#define MVLOG_UNIT_NAME watchdog
#include "XLinkLog.h"
#include "XLink.h"
#include "XLinkPrivateDefines.h"
#include "XLinkErrorUtils.h"

#if defined(_WIN32)
#include "win_synchapi.h"
#endif // defined(_WIN32)

namespace {

using namespace std;
using namespace chrono;
using namespace Watchdog;

/**
 * @brief implementation of watchdog device using xlink representation of it
 */
class XLinkDevice : public IDevice {
    _devicePrivate_t privateDevice;
    using time_point = std::chrono::steady_clock::time_point;
    time_point lastPongTime = time_point::min();
    time_point lastPingTime = time_point::min();
    enum : int { deviceHangTimeout = 12000};

public:
    explicit XLinkDevice(devicePrivate_t *pDevice)
        : privateDevice(*pDevice) {
        setInterval(milliseconds(privateDevice.wd_interval));
    }

    void setInterval(const std::chrono::milliseconds msInterval) noexcept override {
        privateDevice.wd_interval = std::max(static_cast<int>(msInterval.count()), WATCHDOG_PING_INTERVAL_MS);
    }

    void keepAlive(const time_point &current_time) noexcept override {
        bool bPong = sendPingMessage();
        // we consider that as first pong time even if it wasn't happen as beginning of boot
        if (lastPongTime == time_point::min()) {
            lastPongTime = current_time;
        }

        lastPingTime = current_time;

        int diff = duration_cast<milliseconds>(current_time - lastPongTime).count();

        if (bPong) {
            lastPongTime = current_time;
            mvLog(MVLOG_INFO, "[%p] device, ping succeed after %d ms\n", privateDevice.xlink, diff);
        } else {
            mvLog(MVLOG_WARN, "[%p] device, no response for %d ms\n", privateDevice.xlink, diff);
        }
    }

    milliseconds dueIn(const time_point &current_time) const noexcept override {
        if (lastPingTime == time_point::min())
            return milliseconds::zero();

        // overdue
        if (current_time - lastPingTime > std::chrono::milliseconds(privateDevice.wd_interval)) {
            return milliseconds::zero();
        }

        return duration_cast<milliseconds>(lastPingTime + std::chrono::milliseconds(privateDevice.wd_interval) - current_time);
    }

    /**
     * @brief means device is hanging
     */
    bool isTimeout() const noexcept override {
        if (lastPongTime > lastPingTime) return false;
        if (lastPingTime - lastPongTime > milliseconds(deviceHangTimeout)) {
            // cleaning xlink connection - allowing abort all semaphores waiting in other threads
            XLinkResetAll();
            return true;
        }
        return false;
    }

    /**
     * @brief gets some opaque handle that clearly destinguesh one device previate_t from another
     */
    void *getHandle() const noexcept override {
        return privateDevice.xlink;
    }

private:
    bool sendPingMessage() {
        XLinkError_t rc = X_LINK_SUCCESS;
        XLINK_RET_ERR_IF(pthread_mutex_lock(&privateDevice.dev_stream_m), false);

        deviceCommand_t config = {};
        config.type = DEVICE_WATCHDOG_PING;

        // xlink ping acknowledge interval shouldn't be more then expected ping interval
        rc = XLinkWriteDataWithTimeout(privateDevice.device_mon_stream_id, (const uint8_t*)&config, sizeof(config), deviceHangTimeout);

        if(pthread_mutex_unlock(&privateDevice.dev_stream_m) != 0) {
            mvLog(MVLOG_ERROR, "Failed to unlock privateDevice.dev_stream_m");
        }

        if (rc != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Failed send ping message: %s", XLinkErrorToStr(rc));
            return false;
        }
        return true;
    }
};

/**
 * @brief when device just added into watchdog, it should not be due interval at all
 */
class NoDueOnFirstCall : public IDevice {
    std::shared_ptr<IDevice> original;
    bool bFirstCall = false;
 public:
    NoDueOnFirstCall(const std::shared_ptr<IDevice> & original) : original(original) {}
    void setInterval(const std::chrono::milliseconds msInterval) noexcept override {
        original->setInterval(msInterval);
    }
    void keepAlive(const time_point &current_time) noexcept override  {
        original->keepAlive(current_time);
        bFirstCall = true;
    }
    std::chrono::milliseconds dueIn(const time_point &current_time) const noexcept override {
        if (!bFirstCall) {
            return milliseconds::zero();
        }
        return original->dueIn(current_time);
    }
    bool isTimeout() const noexcept override {
        return original->isTimeout();
    }
    void *getHandle() const noexcept override {
        return original->getHandle();
    }
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

static void * WD_OPAQUE_MAGIC = reinterpret_cast<void*>(0xdeadbeaf);

struct wd_context_opaque {
    void * magic = WD_OPAQUE_MAGIC;
    IDevice * actual = nullptr;
    bool   destroyed = false;
    void *handleCached = nullptr;
};

class WatchdogImpl {
    using wd_context_as_tuple = std::tuple<std::shared_ptr<IDevice>, bool*, void*>;

    using Devices = std::list<wd_context_as_tuple>;
    Devices watchedDevices;
    std::atomic_bool threadRunning {false};

    pthread_mutex_t routineLock;
    pthread_cond_t  wakeUpPingThread;
    std::thread poolThread;

    WatchdogImpl(const WatchdogImpl&) = delete;
    WatchdogImpl(WatchdogImpl&&) = delete;
    WatchdogImpl& operator = (const WatchdogImpl&) = delete;
    WatchdogImpl& operator = (WatchdogImpl&&) = delete;

    class AutoScope {
    public:
        explicit AutoScope(const std::function<void()>& func) : _func(func) {}
        ~AutoScope() { _func(); }

        AutoScope(const AutoScope&) = delete;
        AutoScope& operator=(const AutoScope&) = delete;
    private:
        std::function<void()> _func;
    };

private:

    WatchdogImpl() {
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

public:

    static WatchdogImpl &instance() {
        static WatchdogImpl watchdog;
        return watchdog;
    }


    ~WatchdogImpl() {
        mvLog(MVLOG_INFO, "watchdog terminated\n");
        try
        {
            CustomUniqueLock lock {&routineLock};
            for (auto &item : watchedDevices) {
                *std::get<1>(item) = true;
                mvLog(MVLOG_WARN, "[%p] device, stop watching due to watchdog termination\n", std::get<2>(item));
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

        rc = pthread_mutex_destroy(&routineLock);
        if (rc != 0) {
            mvLog(MVLOG_WARN, "failed to destroy the \"routineLock\". rc=%d", rc);
        }

        rc = pthread_cond_destroy(&wakeUpPingThread);
        if (rc != 0) {
            mvLog(MVLOG_WARN, "failed to destroy the \"wakeUpPingThread\". rc=%d", rc);
        }

        if (poolThread.joinable()) {
            poolThread.join();
        }
    }

public:
    void *register_device(std::shared_ptr<IDevice> device) {
        CustomUniqueLock lock {&routineLock};
        std::unique_ptr<wd_context_opaque> ctx (new wd_context_opaque);

        // rare case of exact pointer address collision
        if (ctx.get() == WD_OPAQUE_MAGIC) {
            std::unique_ptr<wd_context_opaque> ctx2(new wd_context_opaque);
            ctx.reset(ctx2.release());
        }

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
                watchdog_routine();
            });
        } else {
            // wake up thread
            int rc = pthread_cond_broadcast(&wakeUpPingThread);
            if (rc != 0) {
                mvLog(MVLOG_WARN, "failed to unblock threads blocked on the \"wakeUpPingThread\". rc=%d", rc);
            }
        }

        ctx->handleCached = device->getHandle();
        watchedDevices.emplace_back(device, &ctx->destroyed, ctx->handleCached);

        ctx->actual = std::get<0>(watchedDevices.back()).get();

        return ctx.release();
    }

    void *register_device(devicePrivate_t *device) {
        return register_device(std::make_shared<NoDueOnFirstCall>(std::make_shared<XLinkDevice>(device)));
    }

    bool remove_device(void *opaque) {
        mvLog(MVLOG_INFO, "remove_device : %p\n", opaque);
        auto ptr  = reinterpret_cast<wd_context_opaque *>(opaque);
        if (ptr == nullptr) {
            return false;
        }

        bool bFound = false;
        {
            CustomUniqueLock lock {&routineLock};

            // thread already removed
            if (ptr->destroyed) {
                delete ptr;
                return true;
            }

            auto idx = std::find_if(std::begin(watchedDevices),
                                    std::end(watchedDevices),
                                    [ptr](const wd_context_as_tuple &item) {
                                        return std::get<0>(item)->getHandle() == ptr->actual->getHandle();
                                    });
            bFound = idx != std::end(watchedDevices);
            if(bFound) {
                watchedDevices.erase(idx);
                delete ptr;
            }
        }

        // wake up thread since we might select removed device as nex to be ping, and there is no more devices available
        int rc = pthread_cond_broadcast(&wakeUpPingThread);
        if (rc != 0) {
            mvLog(MVLOG_WARN, "failed to unblock threads blocked on the \"wakeUpPingThread\". rc=%d", rc);
        }

        return bFound;
    }

 private:
    /// @note: We are using here pthread_cond_timedwait as a replacement for condition_variable::wait_for,
    /// as libstdc++ has bug not using monotonic clock. When GCC 10.x became minimum supported version,
    /// that code could be removed.
    void wait_for(const milliseconds sleepInterval) {
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

    void watchdog_routine() noexcept {
        try {
            mvLog(MVLOG_INFO, "thread started\n");

            milliseconds sleepInterval;

            CustomUniqueLock lock {&routineLock};

            do {
                for (auto deviceIt = watchedDevices.begin(); deviceIt != watchedDevices.end(); ) {
                    auto &device = std::get<0>(*deviceIt);
                    auto isReady = device->dueIn(steady_clock::now()).count() == 0;
                    if (isReady) {
                        auto now = high_resolution_clock::now();
                        device->keepAlive(steady_clock::now());
                        mvLog(MVLOG_DEBUG, "ping completed in %ld ms\n", duration_cast<std::chrono::milliseconds>(high_resolution_clock ::now()-now).count());
                    }
                    if (device->isTimeout()) {
                        mvLog(MVLOG_ERROR, "[%p] device, not respond, removing from watchdog\n", device->getHandle());
                        // marking device as deleted, to prevent double resource free from wd_unregister_device
                        *std::get<1>(*deviceIt) = true;
                        deviceIt = watchedDevices.erase(deviceIt);
                    }
                    else {
                        ++deviceIt;
                    }
                }
                auto currentTime = steady_clock::now();
                auto minInterval = std::min_element(watchedDevices.begin(),
                                                    watchedDevices.end(),
                                                    [&currentTime] (const Devices::value_type & device1, const Devices::value_type & device2) {
                                                        return std::get<0>(device1)->dueIn(currentTime).count()
                                                            < std::get<0>(device2)->dueIn(currentTime).count();
                                                    });
                // if for some reason we have empty devices list but watchdog is active
                if (minInterval == watchedDevices.end()) {
                    mvLog(MVLOG_INFO, "no active devices to watch, stopping  Watchdog thread\n");
                    threadRunning = false;
                    break;
                }
                // TODO: no timer coalescing feature, to minimized thread wakes
                sleepInterval = std::get<0>(*minInterval)->dueIn(currentTime);
                if (sleepInterval.count() <= 0)
                    continue;

                mvLog(MVLOG_DEBUG, "sleep interval = %ld ms\n", sleepInterval.count());
                wait_for(sleepInterval);

                mvLog(MVLOG_DEBUG, "waiting completed in  %ld ms\n",
                      duration_cast<std::chrono::milliseconds>(steady_clock::now() - currentTime).count());
            } while (threadRunning);
        } catch (const std::exception & ex) {
            mvLog(MVLOG_ERROR, "error %s", ex.what());
        } catch (...) {
            mvLog(MVLOG_ERROR, "unknown error");
        }

        mvLog(MVLOG_INFO, "thread ended\n");
    }
};

}  // namespace

WD_API wd_error_t watchdog_init_context(wd_context *ctx) {
    try {
        mvLogLevelSet(MVLOG_ERROR);
        mvLogDefaultLevelSet(MVLOG_ERROR);
        if (!ctx) {
            return WD_NOTINITIALIZED;
        }
        // opaque pointer initialized
        if (ctx->opaque == WD_OPAQUE_MAGIC) {
            mvLog(MVLOG_INFO, "watchdog context (%p) already initialized \n", ctx);
        } else {
            ctx->opaque = WD_OPAQUE_MAGIC;
        }
        return WD_ERRNO;
    }  catch (...) {
        mvLog(MVLOG_ERROR, "failed initialize watchdog context: %p\n", ctx);
    }
    return WD_FAIL;
}

WD_API wd_error_t watchdog_register_device(wd_context * ctx, devicePrivate_t *device) {
    try {
        if (!ctx) {
            mvLog(MVLOG_ERROR, "watchdog context is null\n");
            return WD_NOTINITIALIZED;
        }
        // opaque pointer initialized
        if (ctx->opaque == nullptr) {
            mvLog(MVLOG_ERROR, "watchdog context (%p) not initialized \n", ctx);
            return WD_NOTINITIALIZED;
        }
        if (device && device->wd_interval <= 0) {
            mvLog(MVLOG_ERROR, "watchdog interval should be > 0, but was (%d)\n", device->wd_interval);
            return WD_NOTINITIALIZED;
        }
        // opaque pointer initialized
        if (ctx->opaque != WD_OPAQUE_MAGIC) {
            auto watchee = reinterpret_cast<wd_context_opaque*>(ctx->opaque);
            // NOTE: magic field used to pass preallocated watchee - since this function only used by plugin, this is not a backdoor
            if (watchee->magic == WD_OPAQUE_MAGIC) {
                // actually this can represent already registered context, so need to check
                // since we are adding NoDue wrapper, lets check for it
                if (nullptr != dynamic_cast<NoDueOnFirstCall*>(watchee->actual)) {
                    mvLog(MVLOG_ERROR, "watchdog context (%p) already registered within watchdog\n", ctx);
                    return WD_DUPLICATE;
                }

                // transferring interval from context
                if (device) {
                    watchee->actual->setInterval(milliseconds(device->wd_interval));
                }
                ctx->opaque = WatchdogImpl::instance().register_device(
                    shared_ptr<IDevice>(new NoDueOnFirstCall(shared_ptr<IDevice>(watchee->actual, [](IDevice*){}))));

                if (ctx->opaque == nullptr) {
                    mvLog(MVLOG_ERROR, "watchdog context (%p) not initialized \n", ctx);
                } else {
                    return WD_ERRNO;
                }
            }
            mvLog(MVLOG_ERROR, "watchdog context (%p) not initialized \n", ctx);
            return WD_NOTINITIALIZED;
        }

        if (device && device->wd_interval > 0) {
            ctx->opaque = WatchdogImpl::instance().register_device(device);
        } else {
            ctx->opaque = nullptr;
        }
        return WD_ERRNO;
    } catch (const std::exception & ex) {
        mvLog(MVLOG_ERROR, "failed to register device: %s\n", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "failed to register device context (%p)\n", ctx);
    }
    return WD_FAIL;
}

WD_API wd_error_t watchdog_unregister_device(wd_context *ctx) {
    try {
        if (ctx == nullptr || ctx->opaque == nullptr) {
            return WD_NOTINITIALIZED;
        } else {
            if (ctx->opaque != WD_OPAQUE_MAGIC) {
                auto watchee = reinterpret_cast<wd_context_opaque *>(ctx->opaque);
                // NOTE: magic field used to pass preallocated watchee - since this function only used by plugin, this is not a backdoor
                if (watchee->magic == WD_OPAQUE_MAGIC) {
                    if (!WatchdogImpl::instance().remove_device(ctx->opaque)) {
                        mvLog(MVLOG_WARN, "cannot remove device\n");
                        return WD_FAIL;
                    }
                }
            }
        }

        if (ctx != nullptr) {
            // opaque pointer deleted
            ctx->opaque = nullptr;
        }

        return WD_ERRNO;
    } catch (const std::exception & ex) {
        mvLog(MVLOG_WARN, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_WARN, "unknown error");
    }

    return WD_FAIL;
}
