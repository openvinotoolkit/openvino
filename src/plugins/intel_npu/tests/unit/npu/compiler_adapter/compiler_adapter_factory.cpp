// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "npu/compiler_adapter/fake_vcl.hpp"
#include "npu/fake_backend_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_provider_utils.hpp"

// A test without a device models offline compilation: the compiler-in-plugin is then the only one that can serve the
// request, and the target is given by the platform. The tests named Online* get a device.
namespace {

using namespace ::intel_npu;
using CompilerType = ov::intel_npu::CompilerType;
using ov::test::npu::FakeDevice;
namespace vcl_provider = ov::test::npu::vcl_provider;

// kPluginPlatform and kOtherPluginPlatform are served by the compiler-in-plugin. kDriverOnlyPlatform is served only by
// the driver compiler, when the compilation is online.
constexpr std::string_view kPluginPlatform = ov::intel_npu::Platform::NPU4000;
constexpr std::string_view kDriverOnlyPlatform = ov::intel_npu::Platform::NPU3720;
constexpr std::string_view kOtherPluginPlatform = ov::intel_npu::Platform::NPU5010;

constexpr const char* kFailedProbeMessage = "Failed to create plugin compiler for offline or cross compilation";

CompilerType decide(const CompilerAdapterFactory& factory, std::string_view platform = kPluginPlatform) {
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, nullptr, platform);
    return type;
}

std::shared_ptr<FakeDevice> deviceOf(std::string_view platform) {
    return std::make_shared<FakeDevice>(std::string(platform));
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_AfterPresentOutcome_DoesNotProbeAgain) {
    fake_vcl::FakeVcl fake;
    std::size_t calls = 0;
    const CompilerAdapterFactory factory(vcl_provider::counted(calls, fake.provider()));
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
    }
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_WhenPluginCompilerCannotBeCreated_Throws) {
    const auto expectProbeFailure = [](const VCLFunctionTableProvider& provider) {
        const CompilerAdapterFactory factory(provider);
        OV_EXPECT_THROW_HAS_SUBSTRING(decide(factory), ov::Exception, kFailedProbeMessage);
    };

    // The library cannot be loaded.
    expectProbeFailure(vcl_provider::unavailable());

    // The library loads, but the compiler cannot be created.
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    expectProbeFailure(fake.provider());
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_ExplicitCompilerType_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("An explicit compiler type must not be resolved through the provider"));
    for (const auto explicitType : {CompilerType::DRIVER, CompilerType::PLUGIN}) {
        auto type = explicitType;
        factory.decideCompilerType(type, nullptr, kPluginPlatform);
        EXPECT_EQ(type, explicitType);
    }
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_AfterAbsentOutcome_ThrowsWithoutProbingAgain) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    OV_EXPECT_THROW_HAS_SUBSTRING(decide(factory), ov::Exception, kFailedProbeMessage);
    for (int i = 0; i < 2; i++) {
        OV_EXPECT_THROW_HAS_SUBSTRING(decide(factory), ov::Exception, "Plugin compiler is absent");
    }
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_SeparateFactories_ProbeIndependently) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory first(fake.provider());
    const CompilerAdapterFactory second(fake.provider());
    EXPECT_EQ(decide(first), CompilerType::PLUGIN);
    EXPECT_EQ(decide(second), CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 2u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlinePluginPlatform_WithWorkingCompiler_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kPluginPlatform), kPluginPlatform);

    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineDriverOnlyPlatform_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("Driver only online compilation must not probe the compiler-in-plugin"));
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kDriverOnlyPlatform), kDriverOnlyPlatform);

    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlinePluginPlatform_FallsBackToDriverWhenPluginIsAbsent) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kPluginPlatform), kPluginPlatform);

    EXPECT_EQ(type, CompilerType::DRIVER);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_CrossPlatformWithoutPlugin_Throws) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;

    OV_EXPECT_THROW_HAS_SUBSTRING(factory.decideCompilerType(type, deviceOf(kPluginPlatform), kOtherPluginPlatform),
                                  ov::Exception,
                                  kFailedProbeMessage);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginAfterPresentOutcome_CreatesCompilerWithoutProbing) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    ASSERT_EQ(decide(factory), CompilerType::PLUGIN);

    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler({}, type, kPluginPlatform), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
    // One compiler for the probe of decide(), and one for the adapter that getCompiler() returns.
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 2u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginAfterPresentOutcome_ThrowsWithoutFallback) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    ASSERT_EQ(decide(factory), CompilerType::PLUGIN);

    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    auto type = CompilerType::PREFER_PLUGIN;
    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler({}, type, kPluginPlatform),
                                  ov::Exception,
                                  "VCL compiler loading failed");
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_ExplicitPluginWithFailingCompilerCreation_RetriesAndKeepsOutcomeUnknown) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    for (int i = 0; i < 2; i++) {
        auto type = CompilerType::PLUGIN;
        OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler({}, type, kPluginPlatform),
                                      ov::Exception,
                                      "VCL compiler loading failed");
        EXPECT_EQ(type, CompilerType::PLUGIN);
    }
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 2u);

    // The explicit requests did not decide anything for PREFER_PLUGIN: the next probe sees the fixed compiler.
    fake.succeed(fake_vcl::kCompilerCreate);
    EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_DriverType_ThrowsWithoutInvokingProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("Driver compilation must not request VCL entry points"));
    auto type = CompilerType::DRIVER;
    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler({}, type, kPluginPlatform),
                                  ov::Exception,
                                  "Could not find an NPU device");
}

// Offline compilation can target any platform, including the ones that only the driver compiler serves when the
// compilation is online. Without a device the platform is not looked at, so one platform of each kind is enough.
class CompilerAdapterFactoryPlatformTests : public ::testing::TestWithParam<std::string> {};

TEST_P(CompilerAdapterFactoryPlatformTests, DecideCompilerType_WithWorkingCompiler_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    EXPECT_EQ(decide(factory, GetParam()), CompilerType::PLUGIN);
}

TEST_P(CompilerAdapterFactoryPlatformTests, GetCompiler_PreferPluginFirstCall_ReturnsTheProbedCompiler) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler({}, type, GetParam()), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

INSTANTIATE_TEST_SUITE_P(OfflinePlatforms,
                         CompilerAdapterFactoryPlatformTests,
                         ::testing::Values(std::string(kDriverOnlyPlatform), std::string(kPluginPlatform)),
                         [](const testing::TestParamInfo<std::string>& info) {
                             return "NPU" + info.param;
                         });

}  // namespace
