// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_npu/common/npu.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"

namespace ov::test::npu {

// Hand written fakes of the backend interfaces: they have no call expectations, only canned values and a device lookup
// like the real one.
class FakeDevice final : public ::intel_npu::IDevice {
public:
    static constexpr uint32_t kDeviceId = 0;
    static constexpr uint32_t kSubDevId = 3;
    static constexpr uint32_t kMaxNumSlices = 4;
    static constexpr uint8_t kUuidByte = 0x5A;

    // @param reportsMaxNumSlices When false, getMaxNumSlices() fails like on a device that cannot report it.
    explicit FakeDevice(std::string name, bool reportsMaxNumSlices = true)
        : _name(std::move(name)),
          _reportsMaxNumSlices(reportsMaxNumSlices) {}

    std::string getName() const override {
        return _name;
    }
    std::string getFullDeviceName() const override {
        return "Fake NPU " + _name;
    }
    Uuid getUuid() const override {
        Uuid uuid{};
        uuid.uuid.fill(kUuidByte);
        return uuid;
    }
    uint32_t getSubDevId() const override {
        return kSubDevId;
    }
    uint32_t getMaxNumSlices() const override {
        return _reportsMaxNumSlices ? kMaxNumSlices : ::intel_npu::IDevice::getMaxNumSlices();
    }
    void updateInfo(const ov::AnyMap&) override {}
    bool validateCompatibilityDescriptor(const std::string&) const override {
        return true;
    }
    DeviceProperties getDeviceProperties() const override {
        return {kDeviceId, kSubDevId, kMaxNumSlices};
    }

private:
    std::string _name;
    bool _reportsMaxNumSlices;
};

// What the fake driver reports, handed out through the driver of the fake backend
struct FakeBackendOptions {
    bool contextExtensionSupported = false;
    uint32_t driverVersion = 1;
    uint32_t graphExtVersion = 1;
};

// A driver that reports the values it was given, with a dummy handle.
class FakeDriver final : public ::intel_npu::IDriver {
public:
    explicit FakeDriver(FakeBackendOptions options = {}) : _options(options) {}

    void* getDriverHandle() const override {
        return reinterpret_cast<void*>(0x1);
    }
    uint32_t getDriverVersion() const override {
        return _options.driverVersion;
    }
    uint32_t getGraphExtVersion() const override {
        return _options.graphExtVersion;
    }
    bool isContextExtSupported() const override {
        return _options.contextExtensionSupported;
    }

private:
    const FakeBackendOptions _options;
};

// Mirrors the device lookup of ZeroEngineBackend: an empty id selects the first device, a number is an index when it is
// in range and a name otherwise, and an unknown name throws. No names means no devices, like offline compilation.
class FakeEngineBackend final : public ::intel_npu::IEngineBackend {
public:
    static constexpr const char* kName = "FAKE_BACKEND";

    explicit FakeEngineBackend(const std::vector<std::string>& deviceNames, FakeBackendOptions options = {})
        : _options(options) {
        for (const auto& name : deviceNames) {
            _devices.push_back(std::make_shared<FakeDevice>(name));
        }
    }

    explicit FakeEngineBackend(std::vector<std::shared_ptr<FakeDevice>> devices, FakeBackendOptions options = {})
        : _options(options),
          _devices(std::move(devices)) {}

    const std::shared_ptr<::intel_npu::IDevice> getDevice() const override {
        return _devices.empty() ? nullptr : _devices.front();
    }
    const std::shared_ptr<::intel_npu::IDevice> getDevice(const std::string& name) const override {
        if (_devices.empty()) {
            return nullptr;
        }
        if (name.empty()) {
            return getDevice();
        }
        const auto isDigit = [](unsigned char character) {
            return std::isdigit(character) != 0;
        };
        if (std::all_of(name.begin(), name.end(), isDigit)) {
            const auto index = static_cast<std::size_t>(std::stoul(name));
            if (index < _devices.size()) {
                return _devices[index];
            }
        }
        for (const auto& device : _devices) {
            if (device->getName() == name) {
                return device;
            }
        }
        OPENVINO_THROW("Could not find available NPU device with the specified name: NPU.", name);
    }
    const std::vector<std::string> getDeviceNames() const override {
        std::vector<std::string> names;
        for (const auto& device : _devices) {
            names.push_back(device->getName());
        }
        return names;
    }
    std::shared_ptr<::intel_npu::IDriver> getDriver() const override {
        return std::make_shared<FakeDriver>(_options);
    }
    const std::string getName() const override {
        return kName;
    }
    void updateInfo(const ov::AnyMap& properties) override {
        _updatedProperties = properties;
    }

    // What the last updateInfo() call received.
    const ov::AnyMap& updatedProperties() const {
        return _updatedProperties;
    }

private:
    const FakeBackendOptions _options;
    ov::AnyMap _updatedProperties;
    std::vector<std::shared_ptr<FakeDevice>> _devices;
};

}  // namespace ov::test::npu
