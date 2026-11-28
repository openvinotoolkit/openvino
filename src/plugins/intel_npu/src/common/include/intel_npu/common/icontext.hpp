// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief A driver context that the backend works with.
 * @details The handle is opaque on purpose, so that the common layer does not depend on any driver API. Components
 * that know the driver (for example the level zero ones) use the concrete context to get the typed handle. What the driver
 * supports is described by IDriver.
 */
class IContext {
public:
    virtual ~IContext() = default;

    /**
     * @brief The context handle, nullptr once the context has been destroyed.
     */
    virtual void* getContextHandle() const = 0;

    /**
     * @brief The memory allocated in this context, in bytes. Not every context can report it.
     */
    virtual uint64_t getAllocMemSize() const {
        OPENVINO_THROW("Get AllocMemSize is not supported");
    }
    /**
     * @brief The most memory that can be allocated through this context, in bytes. Not every context can report it.
     */
    virtual uint64_t getTotalMemSize() const {
        OPENVINO_THROW("Get TotalMemSize is not supported");
    }

    /**
     * @brief False once the context has been destroyed, for example after the device was lost.
     */
    bool isAlive() const {
        return getContextHandle() != nullptr;
    }

protected:
    IContext() = default;
    IContext(const IContext&) = default;
    IContext& operator=(const IContext&) = default;
};

}  // namespace intel_npu
