// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace intel_npu {

/**
 * @brief A driver and what it supports.
 * @details The extensions are negotiated with the driver, so their versions are driver facts, shared by every device
 * and context on it. The handle is opaque on purpose, so that the common layer does not depend on any driver API. A
 * driver that cannot provide a value keeps the default, which reads as "not available" (0 or false).
 */
class IDriver {
public:
    virtual ~IDriver() = default;

    virtual void* getDriverHandle() const = 0;

    /** @brief Provide driver version */
    virtual uint32_t getDriverVersion() const {
        return 0;
    }
    /** @brief Provide the API version reported by the driver */
    virtual uint32_t getApiVersion() const {
        return 0;
    }
    /** @brief Provide the version of the graph extension */
    virtual uint32_t getGraphExtVersion() const {
        return 0;
    }
    /** @brief Provide the version of the command queue extension, 0 if not supported */
    virtual uint32_t getCommandQueueExtVersion() const {
        return 0;
    }
    /** @brief The command queue extension is supported (workload type, turbo, ...) */
    bool isCommandQueueExtSupported() const {
        return getCommandQueueExtVersion() != 0;
    }
    /** @brief The driver supports the LUID extension, the value itself belongs to the device */
    virtual bool isLUIDExtSupported() const {
        return false;
    }
    /** @brief The npu context ddi table extension is supported */
    virtual bool isContextExtSupported() const {
        return false;
    }

protected:
    IDriver() = default;
    IDriver(const IDriver&) = default;
    IDriver& operator=(const IDriver&) = default;
};

}  // namespace intel_npu
