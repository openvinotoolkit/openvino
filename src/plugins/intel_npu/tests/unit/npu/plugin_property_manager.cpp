// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_property_manager.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "compiler_adapter/fake_vcl.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "npu/mock_backend_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

using namespace ::intel_npu;
using CompilerType = ov::intel_npu::CompilerType;
using npu_mock::MockDevice;
using npu_mock::MockEngineBackend;

template <class... Options>
void addOptions(OptionsDesc& options) {
    (options.add<Options>(), ...);
}

template <class Action>
void expectExceptionContaining(Action&& action, const std::string& fragment) {
    try {
        action();
        FAIL() << "An exception containing '" << fragment << "' was expected";
    } catch (const ov::Exception& ex) {
        EXPECT_NE(std::string(ex.what()).find(fragment), std::string::npos) << ex.what();
    }
}

bool contains(const std::vector<ov::PropertyName>& properties, const std::string& name) {
    return std::any_of(properties.begin(), properties.end(), [&](const ov::PropertyName& property) {
        return property == name;
    });
}

class PluginPropertyManagerTests : public ::testing::Test {
protected:
    void SetUp() override {
        globalLogLevel = Logger::global().level();
        addOptions<LOG_LEVEL,
                   DEVICE_ID,
                   PLATFORM,
                   COMPILER_TYPE,
                   COMPILER_VERSION,
                   PERF_COUNT,
                   CACHE_DIR,
                   NUM_STREAMS,
                   LOADED_FROM_CACHE,
                   TURBO,
                   TILES,
                   DISABLE_IDLE_MEMORY_PRUNING>(*options);

        backend = makeBackend({"4000"});
        manager = makeManager(backend);
    }

    void TearDown() override {
        Logger::global().setLevel(globalLogLevel);
    }

    ov::SoPtr<IEngineBackend> makeBackend(const std::vector<std::string>& deviceNames) {
        testBackend = std::make_shared<MockEngineBackend>(deviceNames);
        return ov::SoPtr<IEngineBackend>(testBackend);
    }

    // The manager and the option support helper share the one factory of the "plugin".
    std::unique_ptr<PluginPropertyManager> makeManager(const ov::SoPtr<IEngineBackend>& backendToUse) {
        const auto adapterFactory = std::make_shared<const CompilerAdapterFactory>(fake.provider());
        return std::make_unique<PluginPropertyManager>(
            options,
            backendToUse,
            std::make_shared<CompilerOptionSupportHelper>(backendToUse, adapterFactory),
            adapterFactory,
            Logger::global());
    }

    // The manager asks the backend for the Level Zero structures while it registers its properties, so they can only
    // go missing afterwards.
    std::unique_ptr<PluginPropertyManager> makeManagerWithoutLevelZero() {
        auto created = makeManager(backend);
        testBackend->failInitStructs = true;
        return created;
    }

    fake_vcl::FakeVcl fake;
    std::shared_ptr<OptionsDesc> options = std::make_shared<OptionsDesc>();
    std::shared_ptr<MockEngineBackend> testBackend;
    ov::SoPtr<IEngineBackend> backend;
    std::unique_ptr<PluginPropertyManager> manager;
    ov::log::Level globalLogLevel = ov::log::Level::NO;
};

TEST_F(PluginPropertyManagerTests, Construction_WithoutAdapterFactory_Throws) {
    EXPECT_THROW(PluginPropertyManager(options, backend, nullptr, nullptr, Logger::global()), ov::Exception);
}

TEST_F(PluginPropertyManagerTests, SetProperty_RuntimeOption_IsReadBack) {
    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());

    manager->setProperty({{ov::enable_profiling.name(), true}});

    EXPECT_TRUE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetProperty_ValueGivenAsString_IsParsed) {
    manager->setProperty({{ov::enable_profiling.name(), std::string("YES")}});

    EXPECT_TRUE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetProperty_ReadOnlyProperty_Throws) {
    expectExceptionContaining(
        [&] {
            manager->setProperty({{ov::available_devices.name(), std::string("anything")}});
        },
        "READ-ONLY configuration key");
}

TEST_F(PluginPropertyManagerTests, SetProperty_OptionMissingFromTheOptionsDescriptor_Throws) {
    // BATCH_MODE is a known property, but this descriptor does not register its option.
    expectExceptionContaining(
        [&] {
            manager->setProperty({{ov::intel_npu::batch_mode.name(), std::string("AUTO")}});
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, SetProperty_WithAnInvalidKey_AppliesNothing) {
    expectExceptionContaining(
        [&] {
            manager->setProperty({{ov::enable_profiling.name(), true}, {ov::available_devices.name(), std::string{}}});
        },
        "READ-ONLY configuration key");

    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetProperty_LogLevel_PropagatesToTheBackend) {
    manager->setProperty({{ov::log::level.name(), ov::log::Level::ERR}});

    EXPECT_EQ(manager->getProperty(ov::log::level.name()).as<ov::log::Level>(), ov::log::Level::ERR);
    EXPECT_EQ(testBackend->updatedKeys, std::vector<std::string>{ov::log::level.name()});
}

TEST_F(PluginPropertyManagerTests, GetProperty_UnknownKey_Throws) {
    expectExceptionContaining(
        [&] {
            manager->getProperty("NOT_A_PROPERTY");
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, IsPropertySupported_PrivateProperty_IsHiddenButReadable) {
    const auto backendName = ov::intel_npu::backend_name.name();

    EXPECT_FALSE(manager->isPropertySupported(backendName));
    EXPECT_EQ(manager->getProperty(backendName).as<std::string>(), testBackend->getName());
}

TEST_F(PluginPropertyManagerTests, IsPropertySupported_UnknownProperty_IsFalse) {
    EXPECT_FALSE(manager->isPropertySupported("NOT_A_PROPERTY"));
}

TEST_F(PluginPropertyManagerTests, GetProperty_SupportedProperties_ListsPublicSupportedPropertiesWithMutability) {
    const auto supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();

    EXPECT_TRUE(contains(supported, ov::enable_profiling.name()));
    EXPECT_TRUE(contains(supported, ov::available_devices.name()));
    EXPECT_FALSE(contains(supported, ov::intel_npu::backend_name.name()));
    EXPECT_FALSE(contains(supported, ov::loaded_from_cache.name()));

    const auto profiling = std::find(supported.begin(), supported.end(), ov::enable_profiling.name());
    ASSERT_NE(profiling, supported.end());
    EXPECT_TRUE(profiling->is_mutable());
    const auto devices = std::find(supported.begin(), supported.end(), ov::available_devices.name());
    ASSERT_NE(devices, supported.end());
    EXPECT_FALSE(devices->is_mutable());
}

TEST_F(PluginPropertyManagerTests, GetProperty_SupportedProperties_FollowsWhatTheCompilerSupports) {
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::tiles.name()));
    manager = makeManager(backend);
    auto supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_FALSE(contains(supported, ov::intel_npu::tiles.name()));

    fake.unsupportedOptions.clear();
    manager = makeManager(backend);
    supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_TRUE(contains(supported, ov::intel_npu::tiles.name()));
}

TEST_F(PluginPropertyManagerTests, Properties_WithoutBackend_BackendDependentOnesAreUnsupported) {
    manager = makeManager({});

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::driver_version.name()));
    EXPECT_FALSE(manager->isPropertySupported(ov::device::full_name.name()));
    EXPECT_FALSE(manager->isPropertySupported(ov::device::architecture.name()));
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::max_tiles.name()));
    EXPECT_TRUE(manager->getProperty(ov::available_devices.name()).as<std::vector<std::string>>().empty());
}

TEST_F(PluginPropertyManagerTests, GetProperty_AvailableDevices_ListsTheBackendDevices) {
    manager = makeManager(makeBackend({"4000", "5010"}));

    EXPECT_EQ(manager->getProperty(ov::available_devices.name()).as<std::vector<std::string>>(),
              (std::vector<std::string>{"4000", "5010"}));
}

TEST_F(PluginPropertyManagerTests, GetProperty_DeviceFullName_UsesTheDeviceFromTheArguments) {
    manager = makeManager(makeBackend({"4000", "5010"}));

    EXPECT_EQ(manager->getProperty(ov::device::full_name.name()).as<std::string>(),
              MockDevice("4000").getFullDeviceName());
    EXPECT_EQ(manager->getProperty(ov::device::full_name.name(), {{ov::device::id.name(), std::string("5010")}})
                  .as<std::string>(),
              MockDevice("5010").getFullDeviceName());
}

TEST_F(PluginPropertyManagerTests, GetProperty_DeviceFullName_UsesTheDeviceIdFromTheConfig) {
    manager = makeManager(makeBackend({"4000", "5010"}));
    manager->setProperty({{ov::device::id.name(), std::string("5010")}});

    EXPECT_EQ(manager->getProperty(ov::device::full_name.name()).as<std::string>(),
              MockDevice("5010").getFullDeviceName());
}

TEST_F(PluginPropertyManagerTests, IsPropertySupported_DeviceProperty_WithUnknownDevice_IsFalse) {
    const ov::AnyMap unknownDevice{{ov::device::id.name(), std::string("9999")}};

    EXPECT_FALSE(manager->isPropertySupported(ov::device::full_name.name(), unknownDevice));
    expectExceptionContaining(
        [&] {
            manager->getProperty(ov::device::full_name.name(), unknownDevice);
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, GetProperty_DeviceAllocMemSize_ReadsTheDevice) {
    EXPECT_EQ(manager->getProperty(ov::intel_npu::device_alloc_mem_size.name()).as<uint64_t>(),
              MockDevice::kAllocMemSize);
}

TEST_F(PluginPropertyManagerTests, IsPropertySupported_Luid_FollowsTheLuidExtension) {
    EXPECT_FALSE(manager->isPropertySupported(ov::device::luid.name()));

    backend = makeBackend({"4000"});
    testBackend->luidExt = true;
    manager = makeManager(backend);

    EXPECT_TRUE(manager->isPropertySupported(ov::device::luid.name()));
}

TEST_F(PluginPropertyManagerTests, DisableIdleMemoryPruning_WithoutContextExtension_IsUnsupported) {
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::disable_idle_memory_prunning.name()));
    expectExceptionContaining(
        [&] {
            manager->setProperty({{ov::intel_npu::disable_idle_memory_prunning.name(), true}});
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, DisableIdleMemoryPruning_WithContextExtension_UpdatesTheBackend) {
    backend = makeBackend({"4000"});
    testBackend->contextExt = true;
    manager = makeManager(backend);
    ASSERT_TRUE(manager->isPropertySupported(ov::intel_npu::disable_idle_memory_prunning.name()));

    manager->setProperty({{ov::intel_npu::disable_idle_memory_prunning.name(), true}});

    EXPECT_TRUE(manager->getProperty(ov::intel_npu::disable_idle_memory_prunning.name()).as<bool>());
    EXPECT_EQ(testBackend->updatedKeys, std::vector<std::string>{ov::intel_npu::disable_idle_memory_prunning.name()});
}

TEST_F(PluginPropertyManagerTests, Determine_Defaults_ComeFromTheConfig) {
    EXPECT_EQ(manager->determinePlatform({}), ov::intel_npu::Platform::AUTO_DETECT);
    EXPECT_EQ(manager->determineDeviceId({}), "0");
    EXPECT_EQ(manager->determineCompilerType({}), CompilerType::PREFER_PLUGIN);
}

TEST_F(PluginPropertyManagerTests, Determine_Properties_OverrideTheConfig) {
    manager->setProperty({{ov::device::id.name(), std::string("4000")}});

    EXPECT_EQ(manager->determineDeviceId({}), "4000");
    EXPECT_EQ(manager->determineDeviceId({{ov::device::id.name(), std::string("5010")}}), "5010");
    EXPECT_EQ(manager->determinePlatform({{ov::intel_npu::platform.name(), std::string("5010")}}), "5010");
    EXPECT_EQ(manager->determineCompilerType({{ov::intel_npu::compiler_type.name(), std::string("PLUGIN")}}),
              CompilerType::PLUGIN);
}

TEST_F(PluginPropertyManagerTests, DetermineCompilerType_InvalidValue_Throws) {
    EXPECT_THROW(manager->determineCompilerType({{ov::intel_npu::compiler_type.name(), std::string("NOT_A_TYPE")}}),
                 ov::Exception);
}

TEST_F(PluginPropertyManagerTests, SetProperty_CompilerOption_IsAcceptedWhenTheCompilerSupportsIt) {
    manager->setProperty({{"OPT_A", std::string("1")}});

    EXPECT_EQ(manager->getProperty("OPT_A").as<std::string>(), "1");
}

TEST_F(PluginPropertyManagerTests, SetProperty_CompilerOption_IsRejectedWhenTheCompilerDoesNotSupportIt) {
    fake.unsupportedOptions.insert("NOT_A_COMPILER_OPTION");

    expectExceptionContaining(
        [&] {
            manager->setProperty({{"NOT_A_COMPILER_OPTION", std::string("1")}});
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, SetProperty_CompilerOption_UsesTheCompilerTypeOfTheConfig) {
    manager->setProperty({{ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}});

    manager->setProperty({{"OPT_B", std::string("2")}});

    EXPECT_EQ(manager->getProperty("OPT_B").as<std::string>(), "2");
}

TEST_F(PluginPropertyManagerTests, GetProperty_CompilerOptionThatWasNeverSet_Throws) {
    expectExceptionContaining(
        [&] {
            manager->getProperty("OPT_C");
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, IsPropertySupported_CompilerVersion_FollowsTheCompilerCreation) {
    EXPECT_TRUE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));

    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    manager = makeManagerWithoutLevelZero();

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
}

TEST_F(PluginPropertyManagerTests, CompilerQueries_AfterAFailedCompilerCreation_DoNotProbeAgain) {
    fake.failWith("vclCompilerCreate", VCL_RESULT_ERROR_UNKNOWN);
    manager = makeManagerWithoutLevelZero();

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
    EXPECT_THROW(manager->setProperty({{"OPT_A", std::string("1")}}), ov::Exception);
    EXPECT_THROW(manager->getProperty(ov::intel_npu::compiler_version.name()), ov::Exception);

    // Every PREFER_PLUGIN resolution went through the factory of the manager, which remembers the failure.
    EXPECT_EQ(fake.callCount("vclCompilerCreate"), 1u);
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Compile_AppliesKnownOptionsToACopy) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::enable_profiling.name(), true}}, ConfigMergeMode::Compile);

    EXPECT_TRUE(config.get<PERF_COUNT>());
    EXPECT_TRUE(unknown.empty());
    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Compile_KeepsCompilerOptionsAsInternalOnes) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{"OPT_A", std::string("7")}}, ConfigMergeMode::Compile);

    EXPECT_TRUE(config.hasInternal("OPT_A"));
    EXPECT_EQ(config.getInternal("OPT_A"), "7");
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Compile_ForwardsUnknownProperties) {
    fake.unsupportedOptions.insert("FOR_THE_COMPILED_MODEL");

    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{"FOR_THE_COMPILED_MODEL", std::string("1")}, {ov::enable_profiling.name(), true}},
        ConfigMergeMode::Compile);

    EXPECT_EQ(unknown.size(), 1u);
    EXPECT_EQ(unknown.count("FOR_THE_COMPILED_MODEL"), 1u);
    EXPECT_FALSE(config.hasInternal("FOR_THE_COMPILED_MODEL"));
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Query_RejectsUnknownProperties) {
    fake.unsupportedOptions.insert("FOR_THE_COMPILED_MODEL");

    expectExceptionContaining(
        [&] {
            manager->getMergedConfigAndUnknownProperties({{"FOR_THE_COMPILED_MODEL", std::string("1")}},
                                                         ConfigMergeMode::Query);
        },
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_ReadOnlyProperty_Throws) {
    expectExceptionContaining(
        [&] {
            manager->getMergedConfigAndUnknownProperties({{ov::available_devices.name(), std::string{}}},
                                                         ConfigMergeMode::Compile);
        },
        "READ-ONLY configuration key");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Compile_RejectsACompilerPropertyTheCompilerDoesNotSupport) {
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::tiles.name()));

    expectExceptionContaining(
        [&] {
            manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::tiles.name(), int64_t{2}}},
                                                         ConfigMergeMode::Compile);
        },
        "is not supported for current configuration");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Compile_AppliesACompilerPropertyTheCompilerSupports) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::tiles.name(), int64_t{2}}},
                                                     ConfigMergeMode::Compile);

    EXPECT_EQ(config.get<TILES>(), 2);
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Import_SkipsCompileTimeOptions) {
    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{ov::intel_npu::tiles.name(), int64_t{2}}, {ov::enable_profiling.name(), true}},
        ConfigMergeMode::Import);

    EXPECT_TRUE(config.get<PERF_COUNT>());
    EXPECT_NE(config.get<TILES>(), 2);
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Import_ForwardsUnknownPropertiesOfAModelThatWasNotCached) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{"OPT_A", std::string("1")}}, ConfigMergeMode::Import);

    EXPECT_EQ(unknown.count("OPT_A"), 1u);
    EXPECT_FALSE(config.hasInternal("OPT_A"));
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Import_DropsCompilerOptionsOfAModelLoadedFromTheCache) {
    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{"OPT_A", std::string("1")}, {ov::loaded_from_cache.name(), true}},
        ConfigMergeMode::Import);

    EXPECT_TRUE(unknown.empty());
    EXPECT_FALSE(config.hasInternal("OPT_A"));
    EXPECT_TRUE(config.get<LOADED_FROM_CACHE>());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfig_Import_DoesNotKeepTheCompilerType) {
    const auto [compiled, compiledUnknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
                                                     ConfigMergeMode::Compile);
    const auto [imported, importedUnknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
                                                     ConfigMergeMode::Import);

    EXPECT_TRUE(compiled.has(ov::intel_npu::compiler_type.name()));
    EXPECT_FALSE(imported.has(ov::intel_npu::compiler_type.name()));
}

}  // namespace
