// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>

#include "fake_vcl.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace {

using ::intel_npu::CompilerAdapterFactory;
using ::intel_npu::VCLFunctionTable;
using CompilerType = ov::intel_npu::CompilerType;

class FactoryDevice final : public ::intel_npu::IDevice {
public:
    std::string getName() const override {
        return "4000";
    }
    std::string getFullDeviceName() const override {
        return "Factory test device";
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
};

TEST(CompilerAdapterFactoryTests, FailedProviderDoesNotPoisonIndependentHealthyProvider) {
    const CompilerAdapterFactory unavailable([]() -> std::shared_ptr<const VCLFunctionTable> {
        OPENVINO_THROW("Injected unavailable provider");
    });
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_THROW(unavailable.decideCompilerType(type, nullptr, "4000"), ov::Exception);

    fake_vcl::FakeVcl fake;
    size_t calls = 0;
    const CompilerAdapterFactory healthy([&] {
        calls++;
        return fake.functions();
    });
    type = CompilerType::PREFER_PLUGIN;
    healthy.decideCompilerType(type, std::make_shared<FactoryDevice>(), "4000");
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, HealthyProviderDoesNotMaskIndependentUnavailableProvider) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory healthy([&] {
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    healthy.decideCompilerType(type, nullptr, "4000");
    ASSERT_EQ(type, CompilerType::PLUGIN);

    size_t calls = 0;
    const CompilerAdapterFactory unavailable([&]() -> std::shared_ptr<const VCLFunctionTable> {
        calls++;
        OPENVINO_THROW("Injected unavailable provider");
    });
    type = CompilerType::PREFER_PLUGIN;
    unavailable.decideCompilerType(type, std::make_shared<FactoryDevice>(), "4000");
    EXPECT_EQ(type, CompilerType::DRIVER);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, CopiesShareUnavailableResultButExplicitPluginRetries) {
    size_t calls = 0;
    const CompilerAdapterFactory original([&]() -> std::shared_ptr<const VCLFunctionTable> {
        calls++;
        OPENVINO_THROW("Injected unavailable provider");
    });
    const auto copy = original;
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_THROW(original.decideCompilerType(type, nullptr, "4000"), ov::Exception);
    type = CompilerType::PREFER_PLUGIN;
    EXPECT_THROW(copy.decideCompilerType(type, nullptr, "4000"), ov::Exception);
    EXPECT_EQ(calls, 1u);

    type = CompilerType::PLUGIN;
    EXPECT_THROW(copy.getCompiler({}, type, "4000"), ov::Exception);
    EXPECT_EQ(calls, 2u);
}

TEST(CompilerAdapterFactoryTests, CopiesShareAvailableResult) {
    fake_vcl::FakeVcl fake;
    size_t calls = 0;
    const CompilerAdapterFactory original([&] {
        calls++;
        return fake.functions();
    });
    const auto copy = original;
    auto type = CompilerType::PREFER_PLUGIN;
    original.decideCompilerType(type, nullptr, "4000");
    ASSERT_EQ(type, CompilerType::PLUGIN);
    type = CompilerType::PREFER_PLUGIN;
    copy.decideCompilerType(type, nullptr, "4000");
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, DriverPathDoesNotInvokeProvider) {
    const CompilerAdapterFactory factory([]() -> std::shared_ptr<const VCLFunctionTable> {
        ADD_FAILURE() << "Driver compilation must not request VCL entry points";
        OPENVINO_THROW("Unexpected provider invocation");
    });
    auto type = CompilerType::DRIVER;
    EXPECT_THROW(factory.getCompiler({}, type, "4000"), ov::Exception);
}

}  // namespace
