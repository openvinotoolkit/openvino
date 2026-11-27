// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "intel_npu/common/idriver.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

/**
 * @brief What is known about one device. Inference requests are not created here, see IEngineBackend.
 */
class IDevice : public std::enable_shared_from_this<IDevice> {
public:
    using Uuid = ov::device::UUID;
    struct DeviceProperties {
        uint32_t deviceId;
        uint32_t subdeviceId;
        uint32_t numSlices;
    };

    virtual std::string getName() const = 0;
    virtual std::string getFullDeviceName() const = 0;
    virtual Uuid getUuid() const;
    virtual ov::device::LUID getLUID() const;
    virtual uint32_t getSubDevId() const;
    virtual uint32_t getMaxNumSlices() const;
    virtual ov::device::PCIInfo getPciInfo() const;
    virtual ov::device::Type getDeviceType() const;
    virtual std::map<ov::element::Type, float> getGops() const;
    /** @brief The native device handle, opaque for the code that does not know the driver */
    virtual void* getDeviceHandle() const;
    /** @brief The driver the device is on, and what it supports */
    virtual std::shared_ptr<IDriver> getDriver() const;

    virtual void updateInfo(const ov::AnyMap& properties) = 0;
    virtual bool validateCompatibilityDescriptor(const std::string& compatibilityDescriptor) const = 0;
    virtual DeviceProperties getDeviceProperties() const = 0;

protected:
    virtual ~IDevice() = default;
};

}  // namespace intel_npu
