// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

namespace Watchdog {

/**
 * @brief represents watchdog device interface to be registered within watchdog worker
 */
class IDevice {
 public:
    using time_point = std::chrono::high_resolution_clock::time_point;

    virtual ~IDevice() = default;

    /**
     * @brief depending on implementation watchdog device shouldn't have interval longer than that
     */
    virtual void setInterval(const std::chrono::milliseconds msInterval) noexcept = 0;
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

}  // namespace Watchdog
