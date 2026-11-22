// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/device_helpers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/except.hpp"

namespace {

class MockDevice : public ::intel_npu::IDevice {
public:
    explicit MockDevice(std::string name) : _name(std::move(name)) {}

    std::string getName() const override {
        return _name;
    }
    std::string getFullDeviceName() const override {
        return "Mock NPU " + _name;
    }
    std::shared_ptr<::intel_npu::InferRequest> createInferRequest(
        const std::shared_ptr<const ::intel_npu::ICompiledModel>&,
        const ::intel_npu::Config&) override {
        return nullptr;
    }
    void updateInfo(const ov::AnyMap&) override {}
    bool validateCompatibilityDescriptor(const std::string&) const override {
        return true;
    }
    DeviceProperties getDeviceProperties() const override {
        return {};
    }

private:
    std::string _name;
};

// Mirrors ZeroEngineBackend::getDevice(name) lookup: empty -> first device, number -> index if in range, otherwise
// name.
class MockEngineBackend : public ::intel_npu::IEngineBackend {
public:
    explicit MockEngineBackend(std::vector<std::string> deviceNames) {
        for (auto& name : deviceNames) {
            _devices.push_back(std::make_shared<MockDevice>(std::move(name)));
        }
    }

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
        // A number is an index first, otherwise it is searched as an arch name (e.g. "5010")
        if (std::all_of(name.begin(), name.end(), ::isdigit)) {
            const auto index = static_cast<size_t>(std::stoul(name));
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
    const std::string getName() const override {
        return "MOCK";
    }
    bool isCommandQueueExtSupported() const override {
        return false;
    }
    bool isLUIDExtSupported() const override {
        return false;
    }
    bool isContextExtSupported() const override {
        return false;
    }
    void updateInfo(const ov::AnyMap&) override {}

private:
    std::vector<std::shared_ptr<MockDevice>> _devices;
};

ov::SoPtr<::intel_npu::IEngineBackend> makeBackend(std::vector<std::string> deviceNames) {
    return ov::SoPtr<::intel_npu::IEngineBackend>(std::make_shared<MockEngineBackend>(std::move(deviceNames)));
}

const std::string NPU4000{ov::intel_npu::Platform::NPU4000};
const std::string NPU5010{ov::intel_npu::Platform::NPU5010};

using ::intel_npu::utils::getDeviceArchitecture;

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyWithoutBackend) {
    EXPECT_EQ(getDeviceArchitecture({nullptr}, ""), "");
    EXPECT_EQ(getDeviceArchitecture({nullptr}, NPU4000), "");
}

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyWhenBackendHasNoDevices) {
    const auto backend = makeBackend({});

    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, ""), ""));
    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, NPU4000), ""));
}

TEST(DeviceHelpersGetDeviceArchitectureTests, EmptyDeviceIdUsesFirstDevice) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, ""), NPU4000);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, DeviceIdAsNameReturnsMatchingDevice) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, NPU5010), NPU5010);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, DeviceIdAsIndexReturnsDeviceNameNotIndex) {
    const auto backend = makeBackend({NPU4000, NPU5010});

    EXPECT_EQ(getDeviceArchitecture(backend, "0"), NPU4000);
    EXPECT_EQ(getDeviceArchitecture(backend, "1"), NPU5010);
}

TEST(DeviceHelpersGetDeviceArchitectureTests, ReturnsEmptyForUnknownDevice) {
    const auto backend = makeBackend({NPU4000});

    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, "UNKNOWN"), ""));
    EXPECT_NO_THROW(EXPECT_EQ(getDeviceArchitecture(backend, "7"), ""));
}

}  // namespace
