// Copyright (C) 2018-2019 Intel Corporation
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
#include <XLinkPublicDefines.h>
#include <ncCommPrivate.h>
#include <XLink.h>
#include <mvnc.h>
#include <ncPrivateTypes.h>


#define MVLOG_UNIT_NAME watchdog
#include <mvLog.h>
#include <list>
#define _XLINK_ENABLE_PRIVATE_INCLUDE_
#include <XLinkPrivateDefines.h>

namespace {

using namespace std;
using namespace chrono;
using namespace Watchdog;

/**
 * @brief implementation of watchdog device using xlink representation of it
 */
class XLinkDevice : public IDevice {
    _devicePrivate_t privateDevice;
    using time_point = std::chrono::high_resolution_clock::time_point;
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
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&privateDevice.dev_stream_m), false);

        deviceCommand_t config;
        config.type.c1 = CLASS1_WATCHDOG_PING;
        config.optionClass = NC_OPTION_CLASS1;

        // xlink ping acknowledge interval shouldn't be more then expected ping interval
        rc = XLinkWriteDataWithTimeout(privateDevice.device_mon_stream_id, (const uint8_t*)&config, sizeof(config), deviceHangTimeout);

        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&privateDevice.dev_stream_m));

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

static void * WD_OPAQUE_MAGIC = reinterpret_cast<void*>(0xdeadbeaf);

struct wd_context_opaque {
    void * magic = WD_OPAQUE_MAGIC;
    IDevice * actual = nullptr;
    bool   destroyed = false;
    void *handleCached = nullptr;
};

class WatchdogImpl {
    enum : uint8_t {
        STATE_IDLE = 0,
        INITIATE_THREAD_STOP = 1,
        THREAD_EXITED = 2,
        WAKE_UP_THREAD = 3,
    };

    using wd_context_as_tuple = std::tuple<std::shared_ptr<IDevice>, bool*, void*>;

    using Devices = std::list<wd_context_as_tuple>;
    Devices watchedDevices;
    std::mutex devicesListAcc;
    std::atomic<int> generation = {0};
    std::atomic_bool threadRunning;
    volatile std::uint8_t notificationReason = STATE_IDLE;
    std::condition_variable wakeUpPingThread;

    std::thread poolThread;

    WatchdogImpl() = default;
    WatchdogImpl(const WatchdogImpl&) = delete;
    WatchdogImpl(WatchdogImpl&&) = delete;
    WatchdogImpl& operator = (const WatchdogImpl&) = delete;
    WatchdogImpl& operator = (WatchdogImpl&&) = delete;
 public:

    static WatchdogImpl &instance() {
        static WatchdogImpl watchdog;
        return watchdog;
    }

    ~WatchdogImpl() {
        mvLog(MVLOG_INFO, "watchdog terminated\n");
        {
            auto __lock = lock();
            for (auto &item : watchedDevices) {
                *std::get<1>(item) = true;
                mvLog(MVLOG_WARN, "[%p] device, stop watching due to watchdog termination\n", std::get<2>(item));
            }
            notificationReason = THREAD_EXITED;
        }

        wakeUpPingThread.notify_one();

        if (poolThread.joinable()) {
            poolThread.join();
        }
    }

public:
    void *register_device(std::shared_ptr<IDevice> device) {
        auto __locker = lock();
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
                if (pthread_setname_np(pthread_self(), "WatchdogThread") != 0) {
                    perror("Setting name for watchdog thread failed");
                }
                watchdog_routine();
            });
        } else {
            // wake up thread
            notificationReason = WAKE_UP_THREAD;
            wakeUpPingThread.notify_one();
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
        auto __locker = lock();

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
        bool bFound = idx != std::end(watchedDevices);
        watchedDevices.erase(idx);

        // wake up thread since we might select removed device as nex to be ping, and there is no more devices available
        notificationReason = WAKE_UP_THREAD;
        __locker.unlock();
        wakeUpPingThread.notify_one();

        return bFound;
    }

    void clear() {
        {
            mvLog(MVLOG_INFO, "clear\n");
            auto __locker = lock();
            watchedDevices.clear();
            notificationReason = WAKE_UP_THREAD;
        }
        // wake up thread
        wakeUpPingThread.notify_one();
    }

 private:
    std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(devicesListAcc);
    }

    void watchdog_routine() noexcept {
        try {
            mvLog(MVLOG_INFO, "thread started\n");

            milliseconds sleepInterval;
            auto __locker = lock();
            do {
                for (auto deviceIt = watchedDevices.begin(); deviceIt != watchedDevices.end(); ) {
                    auto &device = std::get<0>(*deviceIt);
                    auto isReady = device->dueIn(high_resolution_clock::now()).count() == 0;
                    if (isReady) {
                        auto now = high_resolution_clock::now();
                        device->keepAlive(high_resolution_clock::now());
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
                auto currentTime = high_resolution_clock::now();
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
                mvLog(MVLOG_DEBUG, "sleep interval = %ld ms\n", sleepInterval.count());

                notificationReason = STATE_IDLE;

                wakeUpPingThread.wait_until(__locker, currentTime + sleepInterval, [this, currentTime]() {
                    mvLog(MVLOG_DEBUG,
                          "waiting for %ld ms\n",
                          duration_cast<std::chrono::milliseconds>(high_resolution_clock::now() - currentTime).count());
                    return notificationReason != STATE_IDLE;
                });

                mvLog(MVLOG_DEBUG, "waiting completed in  %ld ms\n",
                      duration_cast<std::chrono::milliseconds>(high_resolution_clock ::now() - currentTime).count());
            } while (notificationReason != THREAD_EXITED);

        } catch (const std::exception & ex) {
            mvLog(MVLOG_ERROR, "error %s\n", ex.what());
        } catch (...) {
            mvLog(MVLOG_ERROR, "error\n");
        }
        mvLog(MVLOG_INFO, "thread ended\n");
        threadRunning = false;
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
    if (ctx == nullptr || ctx->opaque == nullptr) {
        return WD_NOTINITIALIZED;
    } else {
        if (ctx->opaque != WD_OPAQUE_MAGIC) {
            auto watchee = reinterpret_cast<wd_context_opaque*>(ctx->opaque);
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
}
