// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/compiler_adapter_factory.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "fake_vcl.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/vcl/vcl_api.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace {

using namespace ::intel_npu;
using CompilerType = ov::intel_npu::CompilerType;

class FactoryDevice final : public IDevice {
public:
    explicit FactoryDevice(std::string name = "4000") : _name(std::move(name)) {}

    std::string getName() const override {
        return _name;
    }
    std::string getFullDeviceName() const override {
        return "Factory test device";
    }
    std::shared_ptr<InferRequest> createInferRequest(const std::shared_ptr<const ICompiledModel>&,
                                                     const Config&) override {
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

// The real driver adapter needs a Level Zero stack. Throwing from getInitStructs() marks the driver branch, which
// requests them first, without one.
const std::string kDriverBranchReached = "Driver branch reached";

class FactoryBackend final : public IEngineBackend {
public:
    FactoryBackend(std::shared_ptr<IDevice> device, bool throwOnInitStructs)
        : _device(std::move(device)),
          _throwOnInitStructs(throwOnInitStructs) {}

    const std::shared_ptr<IDevice> getDevice() const override {
        return _device;
    }
    const std::string getName() const override {
        return "FACTORY_TEST";
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
    const std::shared_ptr<ZeroInitStructsHolder> getInitStructs() const override {
        if (_throwOnInitStructs) {
            OPENVINO_THROW(kDriverBranchReached);
        }
        return nullptr;
    }

private:
    std::shared_ptr<IDevice> _device;
    bool _throwOnInitStructs;
};

ov::SoPtr<IEngineBackend> makeBackend(bool withDevice, bool throwOnInitStructs = false) {
    return ov::SoPtr<IEngineBackend>(
        std::make_shared<FactoryBackend>(withDevice ? std::make_shared<FactoryDevice>() : nullptr, throwOnInitStructs));
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineWithAvailableProvider_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    size_t calls = 0;
    CompilerAdapterFactory factory([&] {
        calls++;
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, std::make_shared<FactoryDevice>(), "4000");
    EXPECT_EQ(type, CompilerType::PLUGIN);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineWithUnavailableProvider_FallsBackToDriver) {
    size_t calls = 0;
    CompilerAdapterFactory factory([&]() -> std::shared_ptr<const VCLFunctionTable> {
        calls++;
        OPENVINO_THROW("Injected unavailable provider");
    });
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, std::make_shared<FactoryDevice>(), "4000");
    EXPECT_EQ(type, CompilerType::DRIVER);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OfflineWithAvailableProvider_ResolvesToPlugin) {
    fake_vcl::FakeVcl fake;
    CompilerAdapterFactory factory([&] {
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, nullptr, "4000");
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OfflineWithUnavailableProvider_Throws) {
    CompilerAdapterFactory factory([]() -> std::shared_ptr<const VCLFunctionTable> {
        OPENVINO_THROW("Injected unavailable provider");
    });
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_THROW(factory.decideCompilerType(type, nullptr, "4000"), ov::Exception);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_OnlineDriverPlatform_DoesNotInvokeProvider) {
    CompilerAdapterFactory factory([]() -> std::shared_ptr<const VCLFunctionTable> {
        ADD_FAILURE() << "A driver only platform must not request VCL entry points";
        OPENVINO_THROW("Unexpected provider invocation");
    });
    auto type = CompilerType::PREFER_PLUGIN;
    factory.decideCompilerType(type, std::make_shared<FactoryDevice>("3720"), "3720");
    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, DecideCompilerType_ExplicitCompilerType_DoesNotInvokeProvider) {
    CompilerAdapterFactory factory([]() -> std::shared_ptr<const VCLFunctionTable> {
        ADD_FAILURE() << "An explicit compiler type must not be resolved through the provider";
        OPENVINO_THROW("Unexpected provider invocation");
    });
    for (const auto explicitType : {CompilerType::DRIVER, CompilerType::PLUGIN}) {
        auto type = explicitType;
        factory.decideCompilerType(type, std::make_shared<FactoryDevice>(), "4000");
        EXPECT_EQ(type, explicitType);
    }
}

TEST(CompilerAdapterFactoryTests, GetCompiler_ExplicitPluginWithUnavailableProvider_Throws) {
    size_t calls = 0;
    const CompilerAdapterFactory factory([&]() -> std::shared_ptr<const VCLFunctionTable> {
        calls++;
        OPENVINO_THROW("Injected unavailable provider");
    });
    auto type = CompilerType::PLUGIN;
    EXPECT_THROW(factory.getCompiler({}, type, "4000"), ov::Exception);
    EXPECT_EQ(calls, 1u);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineWithFailingCompilerCreation_FallsBackToDriver) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory([&] {
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    try {
        factory.getCompiler(makeBackend(true, true), type, "4000");
        FAIL() << "The driver branch must be reached, and it throws in this test";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find(kDriverBranchReached), std::string::npos) << ex.what();
    }
    EXPECT_TRUE(fake.called("vclCompilerCreate"));
    EXPECT_EQ(type, CompilerType::DRIVER);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOnlineWithWorkingCompiler_UsesPlugin) {
    fake_vcl::FakeVcl fake;
    const CompilerAdapterFactory factory([&] {
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    EXPECT_NE(factory.getCompiler(makeBackend(true), type, "4000"), nullptr);
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_PreferPluginOfflineWithFailingCompilerCreation_Throws) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory([&] {
        return fake.functions();
    });
    auto type = CompilerType::PREFER_PLUGIN;
    try {
        factory.getCompiler(makeBackend(false, true), type, "4000");
        FAIL() << "Offline compilation has no driver compiler to fall back to";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("VCL compiler loading failed"), std::string::npos) << ex.what();
    }
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_ExplicitPluginWithFailingCompilerCreation_ThrowsWithoutFallback) {
    fake_vcl::FakeVcl fake;
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    const CompilerAdapterFactory factory([&] {
        return fake.functions();
    });
    auto type = CompilerType::PLUGIN;
    try {
        factory.getCompiler(makeBackend(true, true), type, "4000");
        FAIL() << "An explicit PLUGIN request must not fall back to the driver compiler";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find("VCL compiler loading failed"), std::string::npos) << ex.what();
    }
    EXPECT_EQ(type, CompilerType::PLUGIN);
}

TEST(CompilerAdapterFactoryTests, GetCompiler_DriverType_DoesNotInvokeProvider) {
    const CompilerAdapterFactory factory([]() -> std::shared_ptr<const VCLFunctionTable> {
        ADD_FAILURE() << "Driver compilation must not request VCL entry points";
        OPENVINO_THROW("Unexpected provider invocation");
    });
    auto type = CompilerType::DRIVER;
    EXPECT_THROW(factory.getCompiler({}, type, "4000"), ov::Exception);
}

}  // namespace
