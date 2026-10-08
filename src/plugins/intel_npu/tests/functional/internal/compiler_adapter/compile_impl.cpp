// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "plugin_property_manager.hpp"

// Matches the [ INFO ] convention used elsewhere in these tests (see core_integration.cpp).
#define NPU_TEST_LOG std::cout << "[          ] [ LOG ] "

namespace {

std::string toString(const ov::CompilationTarget& target) {
    std::ostringstream oss;
    oss << target;
    return oss.str();
}

// No backend and no device involved: offline_compilation_targets must be answerable from the
// plugin's own platform table alone.
class OfflineCompilationTargetsTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;

    void SetUp() override {
        using namespace ::intel_npu;

        options->add<PLATFORM>();
        options->add<DEVICE_ID>();
        options->add<COMPILER_TYPE>();

        const ov::SoPtr<IEngineBackend> backend;
        propertiesManager = std::make_unique<PluginPropertyManager>(
            options,
            backend,
            std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory()),
            Logger::global());
    }
};

// The property must be advertised as supported even with no backend/device present.
TEST_F(OfflineCompilationTargetsTests, IsSupportedWithNoBackend) {
    bool isSupported = false;
    OV_ASSERT_NO_THROW(isSupported = propertiesManager->isPropertySupported(ov::offline_compilation_targets.name()));
    NPU_TEST_LOG << "isPropertySupported(" << ov::offline_compilation_targets.name() << ") = " << std::boolalpha
                 << isSupported << std::endl;
    ASSERT_TRUE(isSupported);
}

// Every known platform must appear, each carrying at least one PCI device ID.
TEST_F(OfflineCompilationTargetsTests, EnumeratesEveryExportablePlatform) {
    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(ov::offline_compilation_targets.name()));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();

    NPU_TEST_LOG << "offline_compilation_targets returned " << targets.size() << " target(s):" << std::endl;
    for (const auto& target : targets) {
        NPU_TEST_LOG << "  " << toString(target) << std::endl;
    }

    std::vector<std::string> platforms;
    for (const auto& target : targets) {
        ASSERT_FALSE(target.device_ids.empty()) << "Target for platform '" << target.platform << "' has no device IDs";
        platforms.push_back(target.platform);
    }
    for (const auto& expectedPlatform : {ov::intel_npu::Platform::NPU3720,
                                         ov::intel_npu::Platform::NPU4000,
                                         ov::intel_npu::Platform::NPU5010,
                                         ov::intel_npu::Platform::NPU5020,
                                         ov::intel_npu::Platform::NPU6010}) {
        ASSERT_TRUE(std::find(platforms.begin(), platforms.end(), std::string(expectedPlatform)) != platforms.end())
            << "Platform '" << expectedPlatform << "' is missing from the offline compilation targets";
    }
}

TEST_F(OfflineCompilationTargetsTests, IndependentOfConfiguredPlatform) {
    // The query must return the full platform list regardless of the platform currently configured -
    // it enumerates what the compiler can build for, not what the current config happens to select.
    propertiesManager->setProperty({{ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU6010))}});

    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(ov::offline_compilation_targets.name()));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();
    NPU_TEST_LOG << "With NPU_PLATFORM=6010 configured, query still returned " << targets.size() << " target(s)"
                 << std::endl;
    ASSERT_GT(targets.size(), 1u);
}

// Unlike the persisted config above, an NPU_PLATFORM passed as a query argument does narrow the result -
// still answered from the plugin's own table alone, no backend or device involved.
TEST_F(OfflineCompilationTargetsTests, FiltersByPlatformArgument) {
    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(
                            ov::offline_compilation_targets.name(),
                            {{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)}}));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();

    NPU_TEST_LOG << "offline_compilation_targets filtered by NPU_PLATFORM=6010 returned " << targets.size()
                 << " target(s)" << std::endl;
    ASSERT_EQ(targets.size(), 1u);
    ASSERT_EQ(targets.front().platform, std::string(ov::intel_npu::Platform::NPU6010));
}

// No backend and no device involved: the selector only has to rewrite the config, which needs
// neither.
class CompilationTargetSelectorTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;

    void SetUp() override {
        using namespace ::intel_npu;

        options->add<PLATFORM>();
        options->add<DEVICE_ID>();
        options->add<COMPILER_TYPE>();
        options->add<COMPILATION_TARGET>();
        options->add<TILES>();

        const ov::SoPtr<IEngineBackend> backend;
        propertiesManager = std::make_unique<PluginPropertyManager>(
            options,
            backend,
            std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory()),
            Logger::global());
    }
};

// The selector property must be advertised as supported.
TEST_F(CompilationTargetSelectorTests, IsSupported) {
    bool isSupported = false;
    OV_ASSERT_NO_THROW(isSupported = propertiesManager->isPropertySupported(ov::compilation_target.name()));
    NPU_TEST_LOG << "isPropertySupported(" << ov::compilation_target.name() << ") = " << std::boolalpha << isSupported
                 << std::endl;
    ASSERT_TRUE(isSupported);
}

// Reading the selector before it has been set must throw a clear, specific error.
TEST_F(CompilationTargetSelectorTests, ThrowsWhenQueriedBeforeBeingSet) {
    try {
        propertiesManager->getProperty(ov::compilation_target.name());
        FAIL() << "Expected getProperty(" << ov::compilation_target.name() << ") to throw";
    } catch (const ov::Exception& e) {
        NPU_TEST_LOG << "getProperty(" << ov::compilation_target.name() << ") threw as expected: \"" << e.what()
                     << "\"" << std::endl;
        ASSERT_NE(std::string(e.what()).find("was not provided"), std::string::npos);
    }
}

// The platform-injection and platform-conflict decisions now live in Plugin::compile_model, above the
// PluginPropertyManager layer these tests exercise - see Plugin::compile_model in plugin.cpp. What is
// still verifiable here is the contract those decisions rely on: COMPILATION_TARGET merges like any
// other property, and never reaches the compiler under its own key.
TEST_F(CompilationTargetSelectorTests, MergesIntoTheConfigWithoutThrowing) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};
    NPU_TEST_LOG << "Merging compilation_target = " << toString(target) << " with tiles = 1" << std::endl;

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(
        merged = propertiesManager->getMergedConfigAndUnknownProperties(
            {ov::compilation_target(target), ov::intel_npu::tiles(1)},
            ::intel_npu::ConfigMergeMode::Compile));

    NPU_TEST_LOG << "Merged config: compilation_target = "
                 << toString(merged.first.get<::intel_npu::COMPILATION_TARGET>())
                 << ", TILES = " << merged.first.get<::intel_npu::TILES>() << std::endl;
    ASSERT_EQ(merged.first.get<::intel_npu::COMPILATION_TARGET>().platform, std::string(ov::intel_npu::Platform::NPU6010));
    ASSERT_EQ(merged.first.get<::intel_npu::TILES>(), 1);
}

// COMPILATION_TARGET's own key must never show up in the string handed to the compiler.
TEST_F(CompilationTargetSelectorTests, NeverReachesTheCompiler) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(merged = propertiesManager->getMergedConfigAndUnknownProperties(
                            {ov::compilation_target(target)}, ::intel_npu::ConfigMergeMode::Compile));

    // isSupported always answers true, so if COMPILATION_TARGET's RunTime mode did not already exclude it,
    // this would not be what is masking it from the output.
    const auto serializedForCompiler = merged.first.toStringForCompiler([](const std::string&) {
        return true;
    });
    NPU_TEST_LOG << "Serialized for compiler: \"" << serializedForCompiler << "\"" << std::endl;
    ASSERT_EQ(serializedForCompiler.find(ov::compilation_target.name()), std::string::npos);
}

// intel_npu::utils::resolveCompilationTarget is the exact logic Plugin::compile_model relies on to
// turn ov::compilation_target into NPU_PLATFORM. Testing it directly - rather than through
// Plugin::compile_model - avoids needing a real backend, device or compiler.
// With no ov::compilation_target present, the properties map must be left untouched.
TEST(ResolveCompilationTargetTests, NoOpWhenNotPresent) {
    ov::AnyMap properties;
    OV_ASSERT_NO_THROW(::intel_npu::utils::resolveCompilationTarget(properties));
    ASSERT_TRUE(properties.empty());
}

// With no explicit platform already set, the target's platform must be injected as NPU_PLATFORM.
TEST(ResolveCompilationTargetTests, InjectsPlatformWhenAbsent) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU4000), {0x643E}};
    ov::AnyMap properties{ov::compilation_target(target)};

    OV_ASSERT_NO_THROW(::intel_npu::utils::resolveCompilationTarget(properties));

    ASSERT_EQ(properties.at(ov::intel_npu::platform.name()).as<std::string>(),
              std::string(ov::intel_npu::Platform::NPU4000));
}

// An explicit platform that agrees with the target's platform must not throw.
TEST(ResolveCompilationTargetTests, AcceptsMatchingExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU4000), {0x643E}};
    ov::AnyMap properties{ov::compilation_target(target),
                          ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU4000))};

    OV_ASSERT_NO_THROW(::intel_npu::utils::resolveCompilationTarget(properties));
}

// An explicit platform that disagrees with the target's platform must throw a clear conflict error.
TEST(ResolveCompilationTargetTests, ThrowsOnConflictingExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU4000), {0x643E}};
    ov::AnyMap properties{ov::compilation_target(target),
                          ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU6010))};

    try {
        ::intel_npu::utils::resolveCompilationTarget(properties);
        FAIL() << "Expected resolveCompilationTarget to throw on a platform conflict";
    } catch (const ov::Exception& e) {
        NPU_TEST_LOG << "resolveCompilationTarget threw as expected: \"" << e.what() << "\"" << std::endl;
        ASSERT_NE(std::string(e.what()).find("conflicts with"), std::string::npos);
    }
}

// Whether NPU6010's platform resolves to more than one device variant is now a compiler-driven
// question (see ICompilerAdapter::resolve_compilation_target_bundles and its
// VCLCompilerImplTest::ResolveBundles* tests), not something this platform-injection step decides -
// it treats NPU6010 like any other platform.
TEST(ResolveCompilationTargetTests, InjectsPlatformForAMultiSkuPlatformToo) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};
    ov::AnyMap properties{ov::compilation_target(target)};

    OV_ASSERT_NO_THROW(::intel_npu::utils::resolveCompilationTarget(properties));

    ASSERT_EQ(properties.at(ov::intel_npu::platform.name()).as<std::string>(),
              std::string(ov::intel_npu::Platform::NPU6010));
}

}  // namespace
