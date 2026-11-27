// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <level_zero/ze_api.h>
#include <ze_graph_ext.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>

#include "intel_npu/common/idevice.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_driver.hpp"
#include "intel_npu/utils/zero/zero_types.hpp"
#include "openvino/runtime/intel_npu/remote_properties.hpp"

namespace intel_npu {

/**
 * @brief The level zero device. It is the only place that holds the native device handle.
 * @details The device is created by the context but does not refer to it: it keeps the library and the driver (the
 * extension tables and capabilities). Memory queries are not here, they are answered by the context. Everybody else
 * gets the device handle from here.
 */
class ZeroDevice : public IDevice {
public:
    /**
     * @param deviceHandle The native device handle
     * @param driver The driver the device is on, with what it supports and its extension tables
     */
    ZeroDevice(ze_device_handle_t deviceHandle,
               const std::shared_ptr<ZeroDriver>& driver);

    /**
     * @brief The graph properties of the device: which compiler it has. Queried on first use.
     */
    ze_device_graph_properties_t getCompilerProperties();
    /**
     * @brief The version of the compiler of the device, in the major / minor encoding of the driver API.
     */
    uint32_t getCompilerVersion();

    /** @brief The typed native handle, for the level zero code */
    ze_device_handle_t getZeDevice() const {
        return _device_handle;
    }

    std::string getName() const override;
    std::string getFullDeviceName() const override;
    Uuid getUuid() const override;
    ov::device::LUID getLUID() const override;
    uint32_t getSubDevId() const override;
    uint32_t getMaxNumSlices() const override;
    ov::device::PCIInfo getPciInfo() const override;
    std::map<ov::element::Type, float> getGops() const override;
    void* getDeviceHandle() const override;
    std::shared_ptr<IDriver> getDriver() const override;
    ov::device::Type getDeviceType() const override;

    void updateInfo(const ov::AnyMap& properties) override;

    bool validateCompatibilityDescriptor(const std::string& compatibilityDescriptor) const override;

    DeviceProperties getDeviceProperties() const override;

    ZeroDevice& operator=(const ZeroDevice&) = delete;
    ZeroDevice(const ZeroDevice&) = delete;

    ~ZeroDevice() = default;

private:
    std::shared_ptr<ZeroDriver> lockDriver() const;

    // Keeps the driver library loaded for as long as the device exists, declared first so it is released last
    const std::shared_ptr<ZeroApi> _zero_api;
    const ze_device_handle_t _device_handle;
    // The driver owns its devices, so the device only refers back to it
    const std::weak_ptr<ZeroDriver> _driver;

    ze_device_properties_t _device_properties = {};
    ze_pci_ext_properties_t _pci_properties = {};
    ze_device_luid_ext_properties_t _device_luid = {};
    std::map<ov::element::Type, float> _device_gops = {{ov::element::f32, 0.f},
                                                       {ov::element::f16, 0.f},
                                                       {ov::element::bf16, 0.f},
                                                       {ov::element::u8, 0.f},
                                                       {ov::element::i8, 0.f}};

    std::mutex _compiler_mutex;
    std::optional<ze_device_graph_properties_t> _compiler_properties = std::nullopt;

    Logger _log;
};
}  // namespace intel_npu
