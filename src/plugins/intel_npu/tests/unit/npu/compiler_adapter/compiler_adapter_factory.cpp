// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fake_vcl.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "npu/mock_backend_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "vcl_provider_utils.hpp"

namespace {

using namespace ::intel_npu;
using CompilerType = ov::intel_npu::CompilerType;
using npu_mock::MockDevice;

// The real driver adapter needs a Level Zero stack. A backend that fails to provide the structures marks the driver
// branch, which requests them first, without one.
ov::SoPtr<IEngineBackend> makeBackend(bool withDevice, bool failInitStructs = false) {
    const auto backend = std::make_shared<npu_mock::MockEngineBackend>(withDevice ? std::vector<std::string>{"4000"}
                                                                                  : std::vector<std::string>{});
    backend->failInitStructs = failInitStructs;
    return ov::SoPtr<IEngineBackend>(backend);
}

// Online compilation on the "4000" device by default; pass nullptr for offline compilation.
CompilerType decide(const CompilerAdapterFactory& factory,
                    const std::shared_ptr<IDevice>& device = std::make_shared<MockDevice>()) {
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, device, "4000");
    return type;
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineWithWorkingCompiler_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    std::size_t calls = 0;
    const CompilerAdapterFactory factory(vcl_provider::counted(calls, fake.provider()));
    EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineWithUnavailableProvider_FallsBackToDriver) {
    std::size_t calls = 0;
    const CompilerAdapterFactory factory(vcl_provider::counted(calls, vcl_provider::unavailable()));
    EXPECT_EQ(decide(factory), CompilerType::DRIVER);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineWithFailingCompilerCreation_FallsBackToDriver) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    EXPECT_EQ(decide(factory), CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OfflineWithWorkingCompiler_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    EXPECT_EQ(decide(factory, nullptr), CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OfflineWithUnavailableProvider_Throws) {
    const CompilerAdapterFactory factory(vcl_provider::unavailable());
    EXPECT_THROW(decide(factory, nullptr), ov::Exception);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OfflineWithFailingCompilerCreation_Throws) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    EXPECT_THROW(decide(factory, nullptr), ov::Exception);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineDriverPlatform_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("A driver only platform must not request VCL entry points"));
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, std::make_shared<MockDevice>("3720"), "3720");
    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_ExplicitCompilerType_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("An explicit compiler type must not be resolved through the provider"));
    for (const auto explicitType : {CompilerType::DRIVER, CompilerType::PLUGIN}) {
        auto type = explicitType;
        factory.decideCompilerType(type, std::make_shared<MockDevice>(), "4000");
        EXPECT_EQ(type, explicitType);
    }
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_AfterPresentOutcome_DoesNotProbeAgain) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    const auto device = std::make_shared<MockDevice>();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(decide(factory, device), CompilerType::PLUGIN);
    }
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_AfterAbsentOutcome_DoesNotProbeAgain) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    const auto device = std::make_shared<MockDevice>();
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(decide(factory, device), CompilerType::DRIVER);
    }
    EXPECT_THROW(decide(factory, nullptr), ov::Exception);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_SeparateFactories_ProbeIndependently) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory first(fake.provider());
    const CompilerAdapterFactory second(fake.provider());
    const auto device = std::make_shared<MockDevice>();
    EXPECT_EQ(decide(first, device), CompilerType::PLUGIN);
    EXPECT_EQ(decide(second, device), CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 2u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineFirstCall_ReturnsTheProbedCompiler) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler(makeBackend(true), type, "4000"), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineAfterPresentOutcome_CreatesCompilerWithoutProbing) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    ASSERT_EQ(decide(factory), CompilerType::PLUGIN);

    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler(makeBackend(true), type, "4000"), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 2u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineWithFailingCompilerCreation_FallsBackToDriver) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    for (int i = 0; i < 2; i++) {
        auto type = CompilerType::PREFER_PLUGIN;
        try {
            factory.getCompiler(makeBackend(true, true), type, "4000");
            FAIL() << "The driver branch must be reached, and it throws in this test";
        } catch (const ov::Exception& ex) {
            EXPECT_NE(std::string(ex.what()).find(npu_mock::kNoInitStructsMessage), std::string::npos) << ex.what();
        }
        EXPECT_EQ(type, CompilerType::DRIVER);
    }
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOfflineWithFailingCompilerCreation_Throws) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_THROW(factory.getCompiler(makeBackend(false, true), type, "4000"), ov::Exception);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineAfterPresentOutcome_ThrowsWithoutFallback) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory(fake.provider());
    ASSERT_EQ(decide(factory), CompilerType::PLUGIN);

    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    auto type = CompilerType::PREFER_PLUGIN;
    try {
        factory.getCompiler(makeBackend(true, true), type, "4000");
        FAIL() << "A request that resolved to PLUGIN must not fall back to the driver compiler";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("VCL compiler loading failed"), std::string::npos) << ex.what();
    }
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_ExplicitPluginWithUnavailableProvider_Throws) {
    std::size_t calls = 0;
    const CompilerAdapterFactory factory(vcl_provider::counted(calls, vcl_provider::unavailable()));
    auto type = CompilerType::PLUGIN;
    EXPECT_THROW(factory.getCompiler({}, type, "4000"), ov::Exception);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_ExplicitPluginWithFailingCompilerCreation_RetriesAndKeepsOutcomeUnknown) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory(fake.provider());
    for (int i = 0; i < 2; i++) {
        auto type = CompilerType::PLUGIN;
        try {
            factory.getCompiler(makeBackend(true, true), type, "4000");
            FAIL() << "An explicit PLUGIN request must not fall back to the driver compiler";
        } catch (const ov::Exception& ex) {
            EXPECT_NE(std::string(ex.what()).find("VCL compiler loading failed"), std::string::npos) << ex.what();
        }
        EXPECT_EQ(type, CompilerType::PLUGIN);
    }
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 2u);

    // The explicit requests did not decide anything for PREFER_PLUGIN: the next probe sees the fixed compiler.
    fake.succeed("vclCompilerCreate");
    EXPECT_EQ(decide(factory), CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_DriverType_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory(
        vcl_provider::forbidden("Driver compilation must not request VCL entry points"));
    auto type = CompilerType::DRIVER;
    EXPECT_THROW(factory.getCompiler({}, type, "4000"), ov::Exception);
}

}  // namespace
