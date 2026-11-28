// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "intel_npu/common/icontext.hpp"
#include "intel_npu/common/idevice.hpp"
#include "intel_npu/common/idriver.hpp"
#include "intel_npu/config/config.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class IEngineBackend : public std::enable_shared_from_this<IEngineBackend> {
public:
    /** @brief Get device, which can be used for inference. Backend responsible for selection. */
    virtual const std::shared_ptr<IDevice> getDevice() const;
    /** @brief Search for a specific device by name */
    virtual const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const;
    /** @brief Get device, which is configured/suitable for provided params */
    virtual const std::shared_ptr<IDevice> getDevice(const ov::AnyMap& paramMap) const;
    /** @brief Provide a list of names of all devices, with which user can work directly */
    virtual const std::vector<std::string> getDeviceNames() const;
    /** @brief Get name of backend */
    virtual const std::string getName() const = 0;
    /** @brief Register backend-specific options */
    virtual void registerOptions(OptionsDesc& options) const;
    /** @brief Get the driver context the backend works with, nullptr if the backend has none */
    virtual std::shared_ptr<IContext> getContext() const;
    /** @brief Get the driver the backend works by default, and what it supports. nullptr if the backend has none */
    virtual std::shared_ptr<IDriver> getDriver() const;
    /** @brief Update backend and device info */
    virtual void updateInfo(const ov::AnyMap& properties) = 0;

protected:
    virtual ~IEngineBackend() = default;
};

}  // namespace intel_npu
