// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xlink_device.h"
#include "watchdog.h"
#include "watchdogPrivate.hpp"

#include "XLink.h"
#include "XLinkPrivateDefines.h"
#include "XLinkErrorUtils.h"

#include <ncPrivateTypes.h>

#include <algorithm>

namespace {

using namespace std;
using namespace chrono;
using namespace Watchdog;

class XLinkDevice : public IDevice {
public:
    explicit XLinkDevice(devicePrivate_t* pDevice);

    void keepAlive(const time_point& current_time) noexcept override;

    milliseconds dueIn(const time_point& current_time) const noexcept override;
    bool isTimeout() const noexcept override;

    /**
     * @brief gets some opaque handle that clearly distinguish one device private_t from another
     */
    void* getHandle() const noexcept override;

    ~XLinkDevice() = default;

private:
    bool sendPingMessage();

private:
    const int kDeviceHangTimeout = 12000;

    _devicePrivate_t m_devicePrivate;

    time_point m_lastPongTime = time_point::min();
    time_point m_lastPingTime = time_point::min();
};

//----------------- XLinkDevice implementation ---------------------

XLinkDevice::XLinkDevice(devicePrivate_t* pDevice)
    : m_devicePrivate(*pDevice) {
    if (m_devicePrivate.wd_interval <= 0) {
        throw runtime_error(
            "watchdog interval should be > 0, but was " + std::to_string(m_devicePrivate.wd_interval));
    }
    m_devicePrivate.wd_interval = std::max(m_devicePrivate.wd_interval, WATCHDOG_MAX_PING_INTERVAL_MS);
}

void XLinkDevice::keepAlive(const time_point &current_time) noexcept {
    bool bPong = sendPingMessage();
    // we consider that as first pong time even if it wasn't happen as beginning of boot
    if (m_lastPongTime == time_point::min()) {
        m_lastPongTime = current_time;
    }

    m_lastPingTime = current_time;

    int diff = (int)duration_cast<milliseconds>(current_time - m_lastPongTime).count();

    if (bPong) {
        m_lastPongTime = current_time;
        mvLog(MVLOG_INFO, "[%p] device, ping succeed after %d ms\n", m_devicePrivate.xlink, diff);
    } else {
        mvLog(MVLOG_WARN, "[%p] device, no response for %d ms\n", m_devicePrivate.xlink, diff);
    }
}

milliseconds XLinkDevice::dueIn(const time_point& current_time) const noexcept {
    if (m_lastPingTime == time_point::min()) {
        return milliseconds::zero();
    }

    // overdue
    if (current_time - m_lastPingTime > std::chrono::milliseconds(m_devicePrivate.wd_interval)) {
        return milliseconds::zero();
    }

    return duration_cast<milliseconds>(m_lastPingTime +
                std::chrono::milliseconds(m_devicePrivate.wd_interval) - current_time);
}

bool XLinkDevice::isTimeout() const noexcept {
    if (m_lastPongTime > m_lastPingTime) {
        return false;
    }

    if (m_lastPingTime - m_lastPongTime > milliseconds(kDeviceHangTimeout)) {
        // cleaning xlink connection - allowing abort all semaphores waiting in other threads
        XLinkResetAll();
        return true;
    }

    return false;
}

void* XLinkDevice::getHandle() const noexcept {
    return m_devicePrivate.xlink;
}

bool XLinkDevice::sendPingMessage() {
    XLINK_RET_ERR_IF(pthread_mutex_lock(&m_devicePrivate.dev_stream_m), false);

    deviceCommand_t config = {};
    config.type = DEVICE_WATCHDOG_PING;

    // xlink ping acknowledge interval shouldn't be more then expected ping interval
    XLinkError_t rc = XLinkWriteDataWithTimeout(m_devicePrivate.device_mon_stream_id,
            (const uint8_t*)&config, sizeof(config), kDeviceHangTimeout);

    if(pthread_mutex_unlock(&m_devicePrivate.dev_stream_m) != 0) {
        mvLog(MVLOG_ERROR, "Failed to unlock m_devicePrivate.dev_stream_m");
    }

    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Failed send ping message: %s", XLinkErrorToStr(rc));
        return false;
    }

    return true;
}

} // namespace

wd_error_t xlink_device_create(WdDeviceHndl_t** out_deviceHandle, devicePrivate_t* pDevice) {
    if (out_deviceHandle == nullptr || pDevice == nullptr) {
        return WD_NOTINITIALIZED;
    }

    *out_deviceHandle = nullptr;
    auto tmpWdDeviceHndl =
        static_cast<WdDeviceHndl_t*>(malloc(sizeof(WdDeviceHndl_t)));
    if(tmpWdDeviceHndl == nullptr) {
        return WD_FAIL;
    }

    try {
        tmpWdDeviceHndl->m_device = new XLinkDevice(pDevice);
        *out_deviceHandle = tmpWdDeviceHndl;
        return WD_ERRNO;
    } catch (const std::exception& ex) {
        mvLog(MVLOG_ERROR, "error %s", ex.what());
    } catch (...) {
        mvLog(MVLOG_ERROR, "unknown error");
    }

    free(tmpWdDeviceHndl);
    return WD_FAIL;
}

void xlink_device_destroy(WdDeviceHndl_t* deviceHandle) {
    if (deviceHandle == nullptr) {
        return;
    }

    if (deviceHandle->m_device != nullptr) {
        delete(reinterpret_cast<XLinkDevice*>(deviceHandle->m_device));
    }

    free(deviceHandle);
}
