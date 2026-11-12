// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
using ov::test::npu::FakeEngineBackend;
namespace vcl_provider = ov::test::npu::vcl_provider;

// kPluginPlatform and kOtherPluginPlatform are served by the compiler-in-plugin. kDriverOnlyPlatform is served only by
// the driver compiler, when the compilation is online.
constexpr std::string_view kPluginPlatform = ov::intel_npu::Platform::NPU4000;
constexpr std::string_view kDriverOnlyPlatform = ov::intel_npu::Platform::NPU3720;
constexpr std::string_view kOtherPluginPlatform = ov::intel_npu::Platform::NPU5010;

CompilerType decide(const CompilerAdapterFactory& factory, std::string_view platform = kPluginPlatform) {
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, nullptr, platform);
    return type;
}

std::shared_ptr<FakeDevice> deviceOf(std::string_view platform) {
    return std::make_shared<FakeDevice>(std::string(platform));
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeLoadsFunctionsWithoutCreatingCompiler) {
    fake_vcl::FakeVcl fake;
    std::size_t calls = 0;
    const CompilerAdapterFactory factory({}, vcl_provider::counted(calls, fake.provider()));
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
    }
    EXPECT_EQ(calls, 3u);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeWhenProviderFailsThrowsOffline) {
    const CompilerAdapterFactory factory({}, vcl_provider::unavailable());
    OV_EXPECT_THROW_HAS_SUBSTRING(decide(factory), ov::Exception, "Failed to load VCL function table");
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeDoesNotCreateCompiler) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory({}, fake.provider());

    EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeExplicitCompilerTypeDoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        {},
        vcl_provider::forbidden("An explicit compiler type must not be resolved through the provider"));
    for (const auto explicitType : {CompilerType::DRIVER, CompilerType::PLUGIN}) {
        auto type = explicitType;
        factory.decideCompilerType(type, nullptr, kPluginPlatform);
        EXPECT_EQ(type, explicitType);
    }
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeProviderFailuresAreRetried) {
    std::size_t calls = 0;
    const CompilerAdapterFactory factory({}, vcl_provider::counted(calls, vcl_provider::unavailable()));
    for (int i = 0; i < 2; i++) {
        OV_EXPECT_THROW_HAS_SUBSTRING(decide(factory), ov::Exception, "Failed to load VCL function table");
    }
    EXPECT_EQ(calls, 2u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeSeparateFactoriesProbeIndependently) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory first({}, fake.provider());
    const CompilerAdapterFactory second({}, fake.provider());
    EXPECT_EQ(decide(first), CompilerType::PLUGIN);
    EXPECT_EQ(decide(second), CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeOnlinePluginPlatformWithWorkingCompilerResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory({}, fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kPluginPlatform), kPluginPlatform);

    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeOnlineDriverOnlyPlatformDoesNotInvokeProvider) {
    const CompilerAdapterFactory factory({}, VCLFunctionTableProvider{});
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kDriverOnlyPlatform), kDriverOnlyPlatform);

    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeOnlinePluginPlatformFallsBackWhenProviderIsUnavailable) {
    const CompilerAdapterFactory factory({}, vcl_provider::unavailable());
    auto type = CompilerType::PREFER_PLUGIN;

    factory.decideCompilerType(type, deviceOf(kPluginPlatform), kPluginPlatform);

    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerTypeCrossPlatformWithoutPluginThrows) {
    const CompilerAdapterFactory factory({}, vcl_provider::unavailable());
    auto type = CompilerType::PREFER_PLUGIN;

    OV_EXPECT_THROW_HAS_SUBSTRING(factory.decideCompilerType(type, deviceOf(kPluginPlatform), kOtherPluginPlatform),
                                  ov::Exception,
                                  "Failed to load VCL function table");
}

TEST(CompilerAdapterFactoryTests, GetCompilerPreferPluginLoadsFunctionsThenCreatesCompiler) {
    fake_vcl::FakeVcl fake;
    std::size_t providerCalls = 0;
    const CompilerAdapterFactory factory({}, vcl_provider::counted(providerCalls, fake.provider()));

    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler(type, kPluginPlatform), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(providerCalls, 2u);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompilerPreferPluginThrowsWhenCompilerCreationFails) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory({}, fake.provider());
    ASSERT_EQ(decide(factory), CompilerType::PLUGIN);

    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    auto type = CompilerType::PREFER_PLUGIN;
    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler(type, kPluginPlatform),
                                  ov::Exception,
                                  "VCL compiler loading failed");
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompilerExplicitPluginWithFailingCompilerCreationRetriesAndKeepsOutcomeUnknown) {
    fake_vcl::FakeVcl fake;
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory({}, fake.provider());
    for (int i = 0; i < 2; i++) {
        auto type = CompilerType::PLUGIN;
        OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler(type, kPluginPlatform),
                                      ov::Exception,
                                      "VCL compiler loading failed");
        EXPECT_EQ(type, CompilerType::PLUGIN);
    }
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 2u);

    // Resolving PREFER_PLUGIN only loads the function table and does not create a compiler.
    fake.succeed(fake_vcl::kCompilerCreate);
    EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 2u);
}

TEST(CompilerAdapterFactoryTests, GetCompilerDriverTypeThrowsWithoutInvokingProvider) {
    const CompilerAdapterFactory factory({}, VCLFunctionTableProvider{});
    auto type = CompilerType::DRIVER;
    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler(type, kPluginPlatform),
                                  ov::Exception,
                                  "Could not find an NPU device");
}

TEST(CompilerAdapterFactoryTests, GetCompilerDriverTypeForAnotherPlatformThanTheDeviceThrowsWithoutInvokingProvider) {
    const auto backend =
        std::make_shared<FakeEngineBackend>(std::vector<std::string>{std::string(kDriverOnlyPlatform)});
    const CompilerAdapterFactory factory(
        ov::SoPtr<IEngineBackend>(backend),
        vcl_provider::forbidden("Driver compilation must not request VCL entry points"));
    auto type = CompilerType::DRIVER;

    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler(type, kPluginPlatform),
                                  ov::Exception,
                                  "Could not find a valid NPU device");
}

TEST(CompilerAdapterFactoryTests, GetCompilerInvalidCompilerTypeThrowsWithoutInvokingProvider) {
    const CompilerAdapterFactory factory(
        ov::SoPtr<IEngineBackend>{nullptr},
        vcl_provider::forbidden("An invalid compiler type must not request VCL entry points"));
    auto type = static_cast<CompilerType>(999);

    OV_EXPECT_THROW_HAS_SUBSTRING(factory.getCompiler(type, kPluginPlatform),
                                  ov::Exception,
                                  "Invalid NPU_COMPILER_TYPE");
}

// Offline compilation can target any platform, including the ones that only the driver compiler serves when the
// compilation is online. Without a device the platform is not looked at, so one platform of each kind is enough.
class CompilerAdapterFactoryPlatformTests : public ::testing::TestWithParam<std::string> {};

TEST_P(CompilerAdapterFactoryPlatformTests, DecideCompilerTypeWithWorkingCompilerResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory({}, fake.provider());
    EXPECT_EQ(decide(factory, GetParam()), CompilerType::PLUGIN);
}

TEST_P(CompilerAdapterFactoryPlatformTests, GetCompilerPreferPluginFirstCallReturnsTheProbedCompiler) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory({}, fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler(type, GetParam()), nullptr);
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
