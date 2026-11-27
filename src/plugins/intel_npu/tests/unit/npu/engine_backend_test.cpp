// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace {

// Version 1.1 in the major / minor encoding of the driver API
constexpr uint32_t kCommandQueueVersion = (1u << 16) | 1u;

// Implements only what is mandatory, to check what a driver that cannot provide the optional values reports
class MinimalDriver final : public intel_npu::IDriver {
public:
    void* getDriverHandle() const override {
        return nullptr;
    }
};

// A driver that supports the command queue extension and nothing else
class CommandQueueDriver final : public intel_npu::IDriver {
public:
    void* getDriverHandle() const override {
        return reinterpret_cast<void*>(0x1);
    }
    uint32_t getCommandQueueExtVersion() const override {
        return kCommandQueueVersion;
    }
};

class MinimalBackend final : public intel_npu::IEngineBackend {
public:
    const std::string getName() const override {
        return "MINIMAL";
    }
    void updateInfo(const ov::AnyMap&) override {}
};

class BackendWithDriver final : public intel_npu::IEngineBackend {
public:
    const std::string getName() const override {
        return "WITH_DRIVER";
    }
    std::shared_ptr<intel_npu::IDriver> getDriver() const override {
        return std::make_shared<CommandQueueDriver>();
    }
    void updateInfo(const ov::AnyMap&) override {}
};

}  // namespace

TEST(DriverCapabilities, DriverThatDoesNotOverrideReportsNothingAvailable) {
    MinimalDriver driver;

    EXPECT_EQ(driver.getDriverVersion(), 0u);
    EXPECT_EQ(driver.getApiVersion(), 0u);
    EXPECT_EQ(driver.getGraphExtVersion(), 0u);
    EXPECT_EQ(driver.getCommandQueueExtVersion(), 0u);
    EXPECT_FALSE(driver.isCommandQueueExtSupported());
    EXPECT_FALSE(driver.isLUIDExtSupported());
    EXPECT_FALSE(driver.isContextExtSupported());
}

TEST(DriverCapabilities, CommandQueueExtensionIsSupportedWhenItsVersionIsReported) {
    CommandQueueDriver driver;

    EXPECT_EQ(driver.getCommandQueueExtVersion(), kCommandQueueVersion);
    EXPECT_TRUE(driver.isCommandQueueExtSupported());
}

TEST(DriverCapabilities, ReportingOneValueDoesNotReportTheOthers) {
    CommandQueueDriver driver;

    EXPECT_EQ(driver.getDriverVersion(), 0u);
    EXPECT_EQ(driver.getGraphExtVersion(), 0u);
    EXPECT_FALSE(driver.isLUIDExtSupported());
    EXPECT_FALSE(driver.isContextExtSupported());
}

TEST(EngineBackend, BackendThatDoesNotOverrideHasNoDriverAndNoContext) {
    MinimalBackend backend;

    EXPECT_EQ(backend.getDriver(), nullptr);
    EXPECT_EQ(backend.getContext(), nullptr);
}

TEST(DriverOf, NoBackendReportsNothingAvailable) {
    const auto driver = intel_npu::utils::driverOf(ov::SoPtr<intel_npu::IEngineBackend>{nullptr});

    ASSERT_NE(driver, nullptr);
    EXPECT_EQ(driver->getDriverHandle(), nullptr);
    EXPECT_EQ(driver->getApiVersion(), 0u);
    EXPECT_FALSE(driver->isCommandQueueExtSupported());
}

TEST(DriverOf, BackendWithoutDriverReportsNothingAvailable) {
    const ov::SoPtr<intel_npu::IEngineBackend> backend(std::make_shared<MinimalBackend>());

    const auto driver = intel_npu::utils::driverOf(backend);

    ASSERT_NE(driver, nullptr);
    EXPECT_FALSE(driver->isLUIDExtSupported());
}

TEST(DriverOf, BackendWithDriverGivesItsDriver) {
    const ov::SoPtr<intel_npu::IEngineBackend> backend(std::make_shared<BackendWithDriver>());

    const auto driver = intel_npu::utils::driverOf(backend);

    ASSERT_NE(driver, nullptr);
    EXPECT_TRUE(driver->isCommandQueueExtSupported());
}
