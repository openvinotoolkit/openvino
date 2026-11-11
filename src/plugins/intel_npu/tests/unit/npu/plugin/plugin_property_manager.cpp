// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_property_manager.hpp"

#include <gtest/gtest.h>
#include <ze_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "metadata.hpp"
#include "npu/compiler_adapter/fake_vcl.hpp"
#include "npu/fake_backend_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

// The compiler of the manager is the one of the plugin, which comes from a fake VCL, so the tests need neither a real
// compiler library nor a driver. The backend is null, as BackendsRegistry::getEngineBackend() returns it when it finds
// no usable driver or device (offline compilation), except in the tests that create a FakeEngineBackend.
namespace {

using namespace ::intel_npu;
using CompilerType = ov::intel_npu::CompilerType;
using ov::test::npu::FakeBackendOptions;
using ov::test::npu::FakeDevice;
using ov::test::npu::FakeEngineBackend;

constexpr std::string_view kBackendPlatform = ov::intel_npu::Platform::NPU5010;
constexpr uint32_t kDriverVersion = 1234;

// The strides option needs a driver with this graph extension version, or a later one.
constexpr uint32_t kStridesGraphExtVersion = ZE_MAKE_VERSION(1, 16);

// Reading it as a string fails, like the id of a caller that passes anything else than a string.
struct ThrowsWhenPrinted {};

std::ostream& operator<<(std::ostream&, const ThrowsWhenPrinted&) {
    OPENVINO_THROW("Cannot be printed");
}

// Only the options the tests below use. A property whose option is not registered is unsupported, and the production
// list is file local to plugin.cpp, so there is nothing to keep in sync.
void registerOptions(OptionsDesc& options) {
    options.add<LOG_LEVEL>();
    options.add<COMPILE_LOG_LEVEL>();
    options.add<CACHE_MODE>();
    options.add<DEVICE_ID>();
    options.add<PERF_COUNT>();
    options.add<LOADED_FROM_CACHE>();
    options.add<TILES>();
    options.add<COMPILER_TYPE>();
    options.add<PLATFORM>();
    options.add<STEPPING>();
    options.add<TURBO>();
    options.add<RUN_INFERENCES_SEQUENTIALLY>();
    options.add<SHARED_COMMON_QUEUE>();
    options.add<WS_COMPILE_CALL_NUMBER>();
    options.add<CACHE_ENCRYPTION_CALLBACKS>();
    options.add<MODEL_PTR>();
    options.add<DISABLE_IDLE_MEMORY_PRUNING>();
    options.add<ENABLE_STRIDES_FOR>();

    OPENVINO_SUPPRESS_DEPRECATED_START
    options.add<ENABLE_CPU_PINNING>();
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool contains(const std::vector<ov::PropertyName>& properties, const std::string& name) {
    return std::any_of(properties.begin(), properties.end(), [&](const ov::PropertyName& property) {
        return property == name;
    });
}

std::shared_ptr<FakeEngineBackend> makeBackend(const FakeBackendOptions& options = {}) {
    return std::make_shared<FakeEngineBackend>(std::vector<std::string>{std::string(kBackendPlatform)}, options);
}

FakeBackendOptions withContextExtension() {
    FakeBackendOptions options;
    options.contextExtensionSupported = true;
    return options;
}

FakeBackendOptions withGraphExtVersion(uint32_t version) {
    FakeBackendOptions options;
    options.graphExtVersion = version;
    return options;
}

class PluginPropertyManagerTests : public ::testing::Test {
protected:
    void SetUp() override {
        globalLogLevel = Logger::global().level();
        registerOptions(*options);
        manager = makeManager();
    }

    void TearDown() override {
        Logger::global().setLevel(globalLogLevel);
    }

    // Every manager gets a factory of its own, like the one of a plugin, so it probes the fake compiler on its own.
    std::unique_ptr<PluginPropertyManager> makeManager() {
        return makeManager(noBackend);
    }

    std::unique_ptr<PluginPropertyManager> makeManager(const ov::SoPtr<IEngineBackend>& backend) {
        const auto adapterFactory = std::make_shared<const CompilerAdapterFactory>(fake.provider());
        return std::make_unique<PluginPropertyManager>(
            options,
            backend,
            std::make_shared<CompilerOptionSupportHelper>(backend, adapterFactory),
            adapterFactory,
            Logger::global());
    }

    fake_vcl::FakeVcl fake;
    std::shared_ptr<OptionsDesc> options = std::make_shared<OptionsDesc>();
    ov::SoPtr<IEngineBackend> noBackend{nullptr};
    std::unique_ptr<PluginPropertyManager> manager;
    ov::log::Level globalLogLevel = ov::log::Level::NO;
};

TEST_F(PluginPropertyManagerTests, SetPropertyRuntimeOptionIsReadBack) {
    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());

    manager->setProperty({{ov::enable_profiling.name(), true}});

    EXPECT_TRUE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetPropertyValueGivenAsStringIsParsed) {
    manager->setProperty({{ov::enable_profiling.name(), std::string("YES")}});

    EXPECT_TRUE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetAndGetDeprecatedCpuPinningUsesTheRuntimeConfig) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    EXPECT_FALSE(manager->isPropertySupported(ov::hint::enable_cpu_pinning.name()));
    manager->setProperty({{ov::hint::enable_cpu_pinning.name(), true}});
    EXPECT_TRUE(manager->getProperty(ov::hint::enable_cpu_pinning.name()).as<bool>());
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::hint::enable_cpu_pinning.name(), true}},
                                                     ConfigMergeMode::Import);
    EXPECT_TRUE(config.get<ENABLE_CPU_PINNING>());
    EXPECT_TRUE(unknown.empty());
    OPENVINO_SUPPRESS_DEPRECATED_END
}

TEST_F(PluginPropertyManagerTests, SetPropertyReadOnlyPropertyThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::available_devices.name(), std::string("anything")}}),
                                  ov::Exception,
                                  "READ-ONLY configuration key");
}

// The plugin registers MODEL_PRIORITY and WORKLOAD_TYPE only when there is a backend, and registerOptions() does not.
TEST_F(PluginPropertyManagerTests, SetPropertyOptionThatIsNotRegisteredIsUnsupported) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->getMergedConfigAndUnknownProperties({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}},
                                                     ConfigMergeMode::Compile),
        ov::Exception,
        "is not supported for current configuration");
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, SetPropertyWithAnInvalidKeyAppliesNothing) {
    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->setProperty({{ov::enable_profiling.name(), true}, {ov::available_devices.name(), std::string{}}}),
        ov::Exception,
        "READ-ONLY configuration key");

    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetPropertyLogLevelIsAppliedToTheGlobalLogger) {
    manager->setProperty({{ov::log::level.name(), ov::log::Level::ERR}});

    EXPECT_EQ(manager->getProperty(ov::log::level.name()).as<ov::log::Level>(), ov::log::Level::ERR);
    EXPECT_EQ(Logger::global().level(), ov::log::Level::ERR);
}

TEST_F(PluginPropertyManagerTests, GetPropertyUnknownKeyThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->getProperty("NOT_A_PROPERTY"),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedPrivatePropertyIsHiddenButReadable) {
    const auto cachingWithMmap = ov::internal::caching_with_mmap.name();

    EXPECT_FALSE(manager->isPropertySupported(cachingWithMmap));
    EXPECT_TRUE(manager->getProperty(cachingWithMmap).as<bool>());
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedUnknownPropertyIsFalse) {
    EXPECT_FALSE(manager->isPropertySupported("NOT_A_PROPERTY"));
}

TEST_F(PluginPropertyManagerTests, GetPropertyWriteOnlyEncryptionCallbacksReturnsDefaultCallbacks) {
    manager->setProperty({{ov::cache_encryption_callbacks.name(), ov::EncryptionCallbacks{nullptr, nullptr}}});

    const auto callbacks = manager->getProperty(ov::cache_encryption_callbacks.name()).as<ov::EncryptionCallbacks>();
    EXPECT_FALSE(callbacks.encrypt);
    EXPECT_FALSE(callbacks.decrypt);
}

TEST_F(PluginPropertyManagerTests, GetPropertyStableRuntimeMetadataReturnsValues) {
    EXPECT_EQ(manager->getProperty(ov::device::capabilities.name()).as<std::vector<std::string>>(),
              (std::vector<std::string>{ov::device::capability::FP16,
                                        ov::device::capability::INT8,
                                        ov::device::capability::EXPORT_IMPORT}));

    const auto asyncRange =
        manager->getProperty(ov::range_for_async_infer_requests.name()).as<std::tuple<uint32_t, uint32_t, uint32_t>>();
    EXPECT_EQ(asyncRange, std::make_tuple(1u, 8u, 1u));

    const auto streamsRange = manager->getProperty(ov::range_for_streams.name()).as<std::tuple<uint32_t, uint32_t>>();
    EXPECT_EQ(streamsRange, std::make_tuple(0u, 8u));
}

TEST_F(PluginPropertyManagerTests, GetPropertyInternalMetadataReturnsValues) {
    EXPECT_TRUE(manager->getProperty(ov::internal::caching_with_mmap.name()).as<bool>());
    EXPECT_FALSE(manager->getProperty(ov::internal::cache_header_alignment.name()).empty());

    const auto internalProperties =
        manager->getProperty(ov::internal::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_TRUE(contains(internalProperties, ov::internal::caching_properties.name()));
    EXPECT_TRUE(contains(internalProperties, ov::internal::cache_header_alignment.name()));

    const auto cachingProperties =
        manager->getProperty(ov::internal::caching_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_TRUE(contains(cachingProperties, ov::cache_mode.name()));
    EXPECT_TRUE(contains(cachingProperties, ov::enable_profiling.name()));
}

TEST_F(PluginPropertyManagerTests, GetPropertyModelAndCompileLogLevelUseOfflineConfig) {
    EXPECT_EQ(manager->getProperty(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>(), nullptr);
    EXPECT_EQ(manager->getProperty(ov::intel_npu::compile_log_level.name()).as<ov::log::Level>(),
              manager->getProperty(ov::log::level.name()).as<ov::log::Level>());

    manager->setProperty({{ov::intel_npu::compile_log_level.name(), std::string("LOG_ERROR")}});
    EXPECT_EQ(manager->getProperty(ov::intel_npu::compile_log_level.name()).as<ov::log::Level>(), ov::log::Level::ERR);
}

TEST_F(PluginPropertyManagerTests, GetPropertySupportedPropertiesListsPublicSupportedPropertiesWithMutability) {
    const auto supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();

    EXPECT_TRUE(contains(supported, ov::enable_profiling.name()));
    EXPECT_TRUE(contains(supported, ov::available_devices.name()));
    EXPECT_FALSE(contains(supported, ov::loaded_from_cache.name()));

    const auto profiling = std::find(supported.begin(), supported.end(), ov::enable_profiling.name());
    ASSERT_NE(profiling, supported.end());
    EXPECT_TRUE(profiling->is_mutable());
    const auto devices = std::find(supported.begin(), supported.end(), ov::available_devices.name());
    ASSERT_NE(devices, supported.end());
    EXPECT_FALSE(devices->is_mutable());
}

TEST_F(PluginPropertyManagerTests, GetPropertySupportedPropertiesFollowsWhatTheCompilerSupports) {
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::tiles.name()));
    manager = makeManager();
    auto supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_FALSE(contains(supported, ov::intel_npu::tiles.name()));

    fake.unsupportedOptions.clear();
    manager = makeManager();
    supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
    EXPECT_TRUE(contains(supported, ov::intel_npu::tiles.name()));
}

TEST_F(PluginPropertyManagerTests, GetPropertySupportedPropertiesOmitsCompilerOptionsWhenCompilerTypeIsEmpty) {
    const auto supported =
        manager->getProperty(ov::supported_properties.name(), {{ov::intel_npu::compiler_type.name(), ov::Any()}})
            .as<std::vector<ov::PropertyName>>();

    EXPECT_FALSE(contains(supported, ov::intel_npu::tiles.name()));
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedPropertiesThatNeedADeviceAreFalse) {
    const std::vector<std::string> needDevice = {ov::intel_npu::driver_version.name(),
                                                 ov::intel_npu::backend_name.name(),
                                                 ov::intel_npu::max_tiles.name(),
                                                 ov::device::full_name.name(),
                                                 ov::device::architecture.name(),
                                                 ov::device::luid.name(),
                                                 ov::compatibility_check.name(),
                                                 ov::intel_npu::disable_idle_memory_prunning.name()};
    const auto supported = manager->getProperty(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();

    for (const auto& name : needDevice) {
        EXPECT_FALSE(manager->isPropertySupported(name)) << name;
        EXPECT_FALSE(contains(supported, name)) << name;
    }
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->getProperty(ov::intel_npu::max_tiles.name()),
                                  ov::Exception,
                                  "Unsupported configuration key");
    EXPECT_TRUE(manager->getProperty(ov::available_devices.name()).as<std::vector<std::string>>().empty());
}

TEST_F(PluginPropertyManagerTests, SetPropertyDisableIdleMemoryPruningIsUnsupported) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::intel_npu::disable_idle_memory_prunning.name(), true}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, SetPropertyDisableIdleMemoryPruningWithContextExtensionUpdatesBackend) {
    const auto backend = makeBackend(withContextExtension());
    manager = makeManager(ov::SoPtr<IEngineBackend>(backend));
    const auto property = ov::intel_npu::disable_idle_memory_prunning.name();

    ASSERT_TRUE(manager->isPropertySupported(property));
    manager->setProperty({{property, true}});

    EXPECT_TRUE(manager->getProperty(property).as<bool>());
    ASSERT_EQ(backend->updatedProperties().count(property), 1u);
    EXPECT_TRUE(backend->updatedProperties().at(property).as<bool>());
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedDisableIdleMemoryPruningWithoutContextExtensionIsFalse) {
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::disable_idle_memory_prunning.name()));
}

TEST_F(PluginPropertyManagerTests, GetPropertyDeviceDerivedValuesUseTheSelectedBackendDevice) {
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));

    EXPECT_EQ(manager->getProperty(ov::device::architecture.name()).as<std::string>(), kBackendPlatform);
    EXPECT_EQ(manager->getProperty(ov::intel_npu::max_tiles.name()).as<int64_t>(),
              static_cast<int64_t>(FakeDevice::kMaxNumSlices));
    EXPECT_EQ(manager->getProperty(ov::intel_npu::stepping.name()).as<int64_t>(),
              static_cast<int64_t>(FakeDevice::kSubDevId));
    EXPECT_FALSE(manager->isPropertySupported(ov::device::architecture.name(),
                                              {{ov::device::id.name(), std::string("missing")}}));
}

TEST_F(PluginPropertyManagerTests, GetPropertyBackendMetadataComesFromTheBackend) {
    FakeBackendOptions options;
    options.driverVersion = kDriverVersion;
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend(options)));

    EXPECT_EQ(manager->getProperty(ov::intel_npu::driver_version.name()).as<uint32_t>(), kDriverVersion);
    EXPECT_EQ(manager->getProperty(ov::intel_npu::backend_name.name()).as<std::string>(), FakeEngineBackend::kName);
    const auto uuid = manager->getProperty(ov::device::uuid.name()).as<ov::device::UUID>();
    EXPECT_TRUE(std::all_of(uuid.uuid.begin(), uuid.uuid.end(), [](uint8_t byte) {
        return byte == FakeDevice::kUuidByte;
    }));
}

TEST_F(PluginPropertyManagerTests, GetPropertyMaxTilesWhenTheDeviceCannotReportThemIsMinusOne) {
    const auto device = std::make_shared<FakeDevice>(std::string(kBackendPlatform), false);
    const auto backend = std::make_shared<FakeEngineBackend>(std::vector<std::shared_ptr<FakeDevice>>{device});
    manager = makeManager(ov::SoPtr<IEngineBackend>(backend));

    EXPECT_EQ(manager->getProperty(ov::intel_npu::max_tiles.name()).as<int64_t>(), -1);
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedDeviceIdThatCannotBeReadAsStringIsFalse) {
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));

    EXPECT_FALSE(
        manager->isPropertySupported(ov::device::architecture.name(), {{ov::device::id.name(), ThrowsWhenPrinted{}}}));
}

TEST_F(PluginPropertyManagerTests, DetermineDefaultsComeFromTheConfig) {
    EXPECT_EQ(manager->determinePlatform({}), ov::intel_npu::Platform::AUTO_DETECT);
    EXPECT_EQ(manager->determineDeviceId({}), "0");
    EXPECT_EQ(manager->determineCompilerType({}), CompilerType::PREFER_PLUGIN);
}

TEST_F(PluginPropertyManagerTests, DeterminePropertiesOverrideTheConfig) {
    const std::string configuredId(ov::intel_npu::Platform::NPU4000);
    const std::string argumentId(ov::intel_npu::Platform::NPU5010);
    const std::string configuredPlatform(ov::intel_npu::Platform::NPU3720);
    manager->setProperty({{ov::device::id.name(), configuredId}, {ov::intel_npu::platform.name(), configuredPlatform}});

    EXPECT_EQ(manager->determineDeviceId({}), configuredId);
    EXPECT_EQ(manager->determineDeviceId({{ov::device::id.name(), argumentId}}), argumentId);
    EXPECT_EQ(manager->getProperty(ov::device::id.name()).as<std::string>(), configuredId);
    EXPECT_EQ(manager->getProperty(ov::device::id.name(), {{ov::device::id.name(), argumentId}}).as<std::string>(),
              argumentId);
    EXPECT_EQ(manager->determinePlatform({}), configuredPlatform);
    EXPECT_EQ(manager->getProperty(ov::intel_npu::platform.name()).as<std::string>(), configuredPlatform);
    EXPECT_EQ(manager->determinePlatform({{ov::intel_npu::platform.name(), argumentId}}), argumentId);
    EXPECT_EQ(manager->determineCompilerType({{ov::intel_npu::compiler_type.name(), std::string("PLUGIN")}}),
              CompilerType::PLUGIN);
    EXPECT_EQ(manager->determineCompilerType({{ov::intel_npu::compiler_type.name(), CompilerType::DRIVER}}),
              CompilerType::DRIVER);
}

TEST_F(PluginPropertyManagerTests, DetermineCompilerTypeInvalidValueThrows) {
    EXPECT_THROW(manager->determineCompilerType({{ov::intel_npu::compiler_type.name(), std::string("NOT_A_TYPE")}}),
                 ov::Exception);
}

TEST_F(PluginPropertyManagerTests, GetPropertyCompilerTypeEmptyArgumentFallsBackToConfiguredValue) {
    EXPECT_EQ(
        manager->getProperty(ov::intel_npu::compiler_type.name(), {{ov::intel_npu::compiler_type.name(), ov::Any()}})
            .as<CompilerType>(),
        CompilerType::PREFER_PLUGIN);
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedCompilerTypeGivenAsStringIsParsedAndInvalidIsUnsupported) {
    EXPECT_EQ(manager
                  ->getProperty(ov::intel_npu::compiler_type.name(),
                                {{ov::intel_npu::compiler_type.name(), std::string("PLUGIN")}})
                  .as<CompilerType>(),
              CompilerType::PLUGIN);
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::tiles.name(),
                                              {{ov::intel_npu::compiler_type.name(), std::string("NOT_A_TYPE")}}));
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name(),
                                              {{ov::intel_npu::compiler_type.name(), std::string("NOT_A_TYPE")}}));
}

TEST_F(PluginPropertyManagerTests, GetPropertySteppingUsesDefaultWithoutBackendAndExplicitValueWhenSet) {
    constexpr int64_t kExplicitStepping = 2;
    EXPECT_EQ(manager->getProperty(ov::intel_npu::stepping.name()).as<int64_t>(), -1);

    manager->setProperty({{ov::intel_npu::stepping.name(), std::to_string(kExplicitStepping)}});

    EXPECT_EQ(manager->getProperty(ov::intel_npu::stepping.name()).as<int64_t>(), kExplicitStepping);
}

TEST_F(PluginPropertyManagerTests, GetPropertyWriteOnlyCompiledModelMetadataReturnsDefaultAndCannotBeSet) {
    EXPECT_FALSE(manager->getProperty(ov::loaded_from_cache.name()).as<bool>());
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::loaded_from_cache.name(), true}}),
                                  ov::Exception,
                                  "only be provided when importing a compiled model");
}

TEST_F(PluginPropertyManagerTests, GetPropertyWorkloadCompileCallNumberThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->getProperty(ov::intel_npu::ws_compile_call_number.name()),
                                  ov::Exception,
                                  "cannot be accessed");
}

TEST_F(PluginPropertyManagerTests, SetPropertyModelOutsideImportThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{ov::hint::model.name(), std::shared_ptr<const ov::Model>{}}}),
                                  ov::Exception,
                                  "can only be provided when importing a compiled model");
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedEnableStridesForWithoutBackendFollowsTheCompiler) {
    EXPECT_TRUE(manager->isPropertySupported(ov::intel_npu::enable_strides_for.name()));

    fake.unsupportedOptions.insert(std::string(ov::intel_npu::enable_strides_for.name()));
    manager = makeManager();
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::enable_strides_for.name()));
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedEnableStridesForFollowsTheGraphExtensionVersion) {
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend(withGraphExtVersion(kStridesGraphExtVersion - 1))));
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::enable_strides_for.name()));

    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend(withGraphExtVersion(kStridesGraphExtVersion))));
    EXPECT_TRUE(manager->isPropertySupported(ov::intel_npu::enable_strides_for.name()));
}

TEST_F(PluginPropertyManagerTests, SetPropertyTurboIsReadBackWhenTheCompilerSupportsIt) {
    manager->setProperty({{ov::intel_npu::turbo.name(), true}});

    EXPECT_TRUE(manager->getProperty(ov::intel_npu::turbo.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, SetPropertyCompilerOptionIsAcceptedWhenTheCompilerSupportsIt) {
    manager->setProperty({{fake_vcl::kOptionA, std::string("1")}});

    EXPECT_EQ(manager->getProperty(fake_vcl::kOptionA).as<std::string>(), "1");
}

TEST_F(PluginPropertyManagerTests, SetPropertyCompilerOptionIsRejectedWhenTheCompilerDoesNotSupportIt) {
    fake.unsupportedOptions.insert("NOT_A_COMPILER_OPTION");

    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{"NOT_A_COMPILER_OPTION", std::string("1")}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, SetPropertyCompilerOptionWithTheDriverCompilerIsRejectedWithoutBackend) {
    manager->setProperty({{ov::intel_npu::compiler_type.name(), CompilerType::DRIVER}});

    OV_EXPECT_THROW_HAS_SUBSTRING(manager->setProperty({{fake_vcl::kOptionA, std::string("1")}}),
                                  ov::Exception,
                                  "Unsupported configuration key");
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 0u);
}

TEST_F(PluginPropertyManagerTests, GetPropertyCompilerOptionThatWasNeverSetThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(manager->getProperty(fake_vcl::kOptionC),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, GetPropertyInternalCompilerOptionWhenCompilerCheckThrowsIsUnsupported) {
    manager->setProperty({{fake_vcl::kOptionA, std::string("1")}});
    manager->setProperty({{ov::intel_npu::compiler_type.name(), CompilerType::DRIVER}});

    OV_EXPECT_THROW_HAS_SUBSTRING(manager->getProperty(fake_vcl::kOptionA),
                                  ov::Exception,
                                  "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, IsPropertySupportedCompilerVersionFollowsTheCompilerCreation) {
    EXPECT_TRUE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
    EXPECT_NO_THROW(manager->getProperty(ov::intel_npu::compiler_version.name()));

    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    manager = makeManager();

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
}

TEST_F(PluginPropertyManagerTests, CompilerQueriesAfterAFailedCompilerCreationDoNotProbeAgain) {
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    manager = makeManager();

    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
    EXPECT_FALSE(manager->isPropertySupported(ov::intel_npu::compiler_version.name()));
    EXPECT_THROW(manager->setProperty({{fake_vcl::kOptionA, std::string("1")}}), ov::Exception);
    EXPECT_THROW(manager->getProperty(ov::intel_npu::compiler_version.name()), ov::Exception);

    // Every PREFER_PLUGIN resolution went through the factory of the manager, which remembers the failure.
    EXPECT_EQ(fake.callCount(fake_vcl::kCompilerCreate), 1u);
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileAppliesKnownOptionsToACopy) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::enable_profiling.name(), true}}, ConfigMergeMode::Compile);

    EXPECT_TRUE(config.get<PERF_COUNT>());
    EXPECT_TRUE(unknown.empty());
    EXPECT_FALSE(manager->getProperty(ov::enable_profiling.name()).as<bool>());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileAppliesCompilerPropertiesTheCompilerSupports) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::turbo.name(), true}}, ConfigMergeMode::Compile);

    EXPECT_TRUE(config.get<TURBO>());
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileRejectsACompilerPropertyTheCompilerDoesNotSupport) {
    fake.unsupportedOptions.insert(std::string(ov::intel_npu::tiles.name()));

    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::tiles.name(), int64_t{2}}},
                                                     ConfigMergeMode::Compile),
        ov::Exception,
        "is not supported for current configuration");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileKeepsCompilerOptionsAsInternalOnes) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{fake_vcl::kOptionA, std::string("7")}},
                                                     ConfigMergeMode::Compile);

    EXPECT_TRUE(config.hasInternal(fake_vcl::kOptionA));
    EXPECT_EQ(config.getInternal(fake_vcl::kOptionA), "7");
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileForwardsUnknownProperties) {
    fake.unsupportedOptions.insert("FOR_THE_COMPILED_MODEL");

    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{"FOR_THE_COMPILED_MODEL", true}, {ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
        ConfigMergeMode::Compile);

    EXPECT_EQ(config.get<COMPILER_TYPE>(), CompilerType::PLUGIN);
    ASSERT_EQ(unknown.size(), 1u);
    EXPECT_TRUE(unknown.at("FOR_THE_COMPILED_MODEL").as<bool>());
    EXPECT_FALSE(config.hasInternal("FOR_THE_COMPILED_MODEL"));
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileKeepsTheRequestedPlatformAndCompilerType) {
    const std::string platform(ov::intel_npu::Platform::NPU3720);
    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{ov::intel_npu::platform.name(), platform}, {ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
        ConfigMergeMode::Compile);

    EXPECT_EQ(config.get<PLATFORM>(), platform);
    EXPECT_EQ(config.get<COMPILER_TYPE>(), CompilerType::PLUGIN);
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigQueryRejectsUnknownProperties) {
    fake.unsupportedOptions.insert("FOR_THE_COMPILED_MODEL");

    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->getMergedConfigAndUnknownProperties({{"FOR_THE_COMPILED_MODEL", std::string("1")}},
                                                     ConfigMergeMode::Query),
        ov::Exception,
        "Unsupported configuration key");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCompileUnknownPropertyWhenCompilerCheckThrowsIsForwarded) {
    manager->setProperty({{ov::intel_npu::compiler_type.name(), CompilerType::DRIVER}});

    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{"DRIVER_ONLY_UNKNOWN", std::string("value")}},
                                                     ConfigMergeMode::Compile);

    EXPECT_EQ(config.get<COMPILER_TYPE>(), CompilerType::DRIVER);
    EXPECT_EQ(unknown.at("DRIVER_ONLY_UNKNOWN").as<std::string>(), "value");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigReadOnlyPropertyThrows) {
    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->getMergedConfigAndUnknownProperties({{ov::available_devices.name(), std::string{}}},
                                                     ConfigMergeMode::Compile),
        ov::Exception,
        "READ-ONLY configuration key");
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportSkipsCompileTimeOptions) {
    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{ov::intel_npu::tiles.name(), int64_t{2}}, {ov::enable_profiling.name(), true}},
        ConfigMergeMode::Import);

    EXPECT_TRUE(config.get<PERF_COUNT>());
    EXPECT_NE(config.get<TILES>(), 2);
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportSkipsUnsupportedBothOptionWithoutCompilerType) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::turbo.name(), true}}, ConfigMergeMode::Import);

    EXPECT_FALSE(config.get<TURBO>());
    EXPECT_TRUE(unknown.empty());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportForwardsUnknownPropertiesOfAModelThatWasNotCached) {
    const auto [config, unknown] =
        manager->getMergedConfigAndUnknownProperties({{fake_vcl::kOptionA, std::string("1")}}, ConfigMergeMode::Import);

    EXPECT_EQ(unknown.count(fake_vcl::kOptionA), 1u);
    EXPECT_FALSE(config.hasInternal(fake_vcl::kOptionA));
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportDropsCompilerOptionsOfAModelLoadedFromTheCache) {
    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{fake_vcl::kOptionA, std::string("1")}, {ov::loaded_from_cache.name(), true}},
        ConfigMergeMode::Import);

    EXPECT_TRUE(unknown.empty());
    EXPECT_FALSE(config.hasInternal(fake_vcl::kOptionA));
    EXPECT_TRUE(config.get<LOADED_FROM_CACHE>());
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportDoesNotKeepTheCompilerType) {
    const auto [compiled, compiledUnknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
                                                     ConfigMergeMode::Compile);
    const auto [imported, importedUnknown] =
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::compiler_type.name(), CompilerType::PLUGIN}},
                                                     ConfigMergeMode::Import);

    EXPECT_TRUE(compiled.has(ov::intel_npu::compiler_type.name()));
    EXPECT_FALSE(imported.has(ov::intel_npu::compiler_type.name()));
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigImportStoresModelPointer) {
    const auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    const auto result = std::make_shared<ov::op::v0::Result>(parameter);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});
    const std::shared_ptr<const ov::Model> constModel = model;

    // The property takes the model as a mutable or as a const pointer.
    for (const ov::Any& modelProperty : {ov::Any(model), ov::Any(constModel)}) {
        const auto [config, unknown] =
            manager->getMergedConfigAndUnknownProperties({{ov::hint::model.name(), modelProperty}},
                                                         ConfigMergeMode::Import);

        EXPECT_TRUE(config.has(ov::hint::model.name()));
        EXPECT_EQ(config.get<MODEL_PTR>().lock(), model);
        EXPECT_TRUE(unknown.empty());
    }
}

TEST_F(PluginPropertyManagerTests, GetMergedConfigCommandQueueOptionsEnforcesOfflineCombinationRules) {
    EXPECT_FALSE(manager->getProperty(ov::intel_npu::run_inferences_sequentially.name()).as<bool>());
    EXPECT_FALSE(manager->getProperty(ov::intel_npu::shared_common_queue.name()).as<bool>());

    const auto [config, unknown] = manager->getMergedConfigAndUnknownProperties(
        {{ov::intel_npu::run_inferences_sequentially.name(), false}, {ov::intel_npu::shared_common_queue.name(), true}},
        ConfigMergeMode::Compile);

    EXPECT_FALSE(config.get<RUN_INFERENCES_SEQUENTIALLY>());
    EXPECT_TRUE(config.get<SHARED_COMMON_QUEUE>());
    EXPECT_TRUE(unknown.empty());

    const auto [independentConfig, independentUnknown] = manager->getMergedConfigAndUnknownProperties(
        {{ov::intel_npu::run_inferences_sequentially.name(), true}, {ov::intel_npu::shared_common_queue.name(), false}},
        ConfigMergeMode::Compile);
    EXPECT_TRUE(independentConfig.get<RUN_INFERENCES_SEQUENTIALLY>());
    EXPECT_FALSE(independentConfig.get<SHARED_COMMON_QUEUE>());
    EXPECT_TRUE(independentUnknown.empty());

    manager->setProperty({{ov::intel_npu::run_inferences_sequentially.name(), true}});
    OV_EXPECT_THROW_HAS_SUBSTRING(
        manager->getMergedConfigAndUnknownProperties({{ov::intel_npu::shared_common_queue.name(), true}},
                                                     ConfigMergeMode::Compile),
        ov::Exception,
        "is not supported for current configuration");
}

// The fake backend has no Level Zero structures, so the compatibility check goes through the compiler of the plugin.
class PluginPropertyManagerCompatibilityCheckTests : public PluginPropertyManagerTests {
protected:
    void SetUp() override {
        PluginPropertyManagerTests::SetUp();
        manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));
    }

    static std::string requirementsWith(std::optional<std::string_view> compatibilityDescriptor) {
        Metadata<METADATA_VERSION_2_6> metadata(0,
                                                CURRENT_OPENVINO_VERSION,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                compatibilityDescriptor);
        std::ostringstream stream;
        metadata.write_as_text(stream);
        return stream.str();
    }

    ov::CompatibilityCheck check(const std::string& runtimeRequirements) const {
        return manager
            ->getProperty(ov::compatibility_check.name(), {{ov::runtime_requirements.name(), runtimeRequirements}})
            .as<ov::CompatibilityCheck>();
    }

    const std::string compatibilityCheck = ov::compatibility_check.name();
    const std::string descriptor = "platform=NPU3720;tiles=2;something=ELSE";

    // The compiler must have been asked about the descriptor, or the outcome comes from somewhere else.
    void expectDescriptorWasQueried() const {
        ASSERT_FALSE(fake.optionSupportQueries.empty());
        EXPECT_EQ(fake.optionSupportQueries.back().first, compatibilityCheck);
        EXPECT_EQ(fake.optionSupportQueries.back().second, descriptor);
    }
};

TEST_F(PluginPropertyManagerCompatibilityCheckTests, IsPropertySupportedFollowsTheCompiler) {
    EXPECT_TRUE(manager->isPropertySupported(compatibilityCheck));

    fake.unsupportedOptions.insert(compatibilityCheck);
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));
    EXPECT_FALSE(manager->isPropertySupported(compatibilityCheck));
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, IsPropertySupportedWhenCompilerCreationFailsIsFalse) {
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);
    manager = makeManager(ov::SoPtr<IEngineBackend>(makeBackend()));

    EXPECT_FALSE(manager->isPropertySupported(compatibilityCheck));
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWithoutRuntimeRequirementsIsNotApplicable) {
    EXPECT_EQ(manager->getProperty(compatibilityCheck).as<ov::CompatibilityCheck>(),
              ov::CompatibilityCheck::NOT_APPLICABLE);
    EXPECT_EQ(manager->getProperty(compatibilityCheck, {{ov::device::id.name(), std::string("0")}})
                  .as<ov::CompatibilityCheck>(),
              ov::CompatibilityCheck::NOT_APPLICABLE);
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWithEmptyRuntimeRequirementsIsNotApplicable) {
    EXPECT_EQ(check(""), ov::CompatibilityCheck::NOT_APPLICABLE);
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWithMalformedRuntimeRequirementsIsUnsupported) {
    EXPECT_EQ(check("not_a_metadata_text"), ov::CompatibilityCheck::UNSUPPORTED);
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWithoutCompatibilityDescriptorIsNotApplicable) {
    EXPECT_EQ(check(requirementsWith(std::nullopt)), ov::CompatibilityCheck::NOT_APPLICABLE);
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyAsksTheCompilerAboutTheDescriptor) {
    ASSERT_TRUE(manager->isPropertySupported(compatibilityCheck));

    EXPECT_EQ(check(requirementsWith(descriptor)), ov::CompatibilityCheck::SUPPORTED);
    expectDescriptorWasQueried();
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWhenTheCompilerRejectsTheDescriptorIsUnsupported) {
    ASSERT_TRUE(manager->isPropertySupported(compatibilityCheck));
    fake.unsupportedOptions.insert(compatibilityCheck);

    EXPECT_EQ(check(requirementsWith(descriptor)), ov::CompatibilityCheck::UNSUPPORTED);
    expectDescriptorWasQueried();
}

TEST_F(PluginPropertyManagerCompatibilityCheckTests, GetPropertyWhenTheCompilerCannotBeCreatedIsNotApplicable) {
    ASSERT_TRUE(manager->isPropertySupported(compatibilityCheck));
    fake.failWith(fake_vcl::kCompilerCreate, VCL_RESULT_ERROR_UNKNOWN);

    EXPECT_EQ(check(requirementsWith(descriptor)), ov::CompatibilityCheck::NOT_APPLICABLE);
}

}  // namespace
