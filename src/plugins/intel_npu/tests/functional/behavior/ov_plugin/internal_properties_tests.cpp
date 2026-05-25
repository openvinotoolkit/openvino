// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal_properties_tests.hpp"

#include <cstdlib>

#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

namespace {

void set_env(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    ::setenv(name.c_str(), value.c_str(), 1);
#endif
}

void unset_env(const std::string& name) {
#ifdef _WIN32
    _putenv_s(name.c_str(), "");
#else
    ::unsetenv(name.c_str());
#endif
}

}  // namespace

namespace ov::test::behavior {

std::string OVPropertiesTestsNPU::getTestCaseName(const testing::TestParamInfo<PropertiesParamsNPU>& obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), " "), "_");
    }
    return result.str();
}

void OVPropertiesTestsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(target_device, properties) = this->GetParam();
    APIBaseTest::SetUp();
    model = ov::test::utils::make_split_concat();
}

void OVPropertiesTestsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

std::string OVPropertiesTestsWithCompileModelPropsNPU::getTestCaseName(
    const testing::TestParamInfo<PropertiesParamsNPU>& obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), " "), "_");
    }
    return result.str();
}

void OVPropertiesTestsWithCompileModelPropsNPU::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string temp_device;
    std::tie(temp_device, properties) = this->GetParam();
    std::string::size_type pos = temp_device.find(":", 0);
    std::string hw_device;

    if (pos == std::string::npos) {
        target_device = temp_device;
        hw_device = temp_device;
    } else {
        target_device = temp_device.substr(0, pos);
        hw_device = temp_device.substr(++pos, std::string::npos);
    }

    if (target_device == std::string(ov::test::utils::DEVICE_MULTI) ||
        target_device == std::string(ov::test::utils::DEVICE_AUTO) ||
        target_device == std::string(ov::test::utils::DEVICE_HETERO) ||
        target_device == std::string(ov::test::utils::DEVICE_BATCH)) {
        compileModelProperties = {ov::device::priorities(hw_device)};
    }

    model = ov::test::utils::make_split_concat();

    APIBaseTest::SetUp();
}

void OVPropertiesTestsWithCompileModelPropsNPU::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

TEST_P(OVPropertiesTestsNPU, SetCorrectProperties) {
    core->get_versions(target_device);
    core->set_property(target_device, properties);
}

TEST_P(OVPropertiesTestsNPU, canSetPropertyAndCheckGetProperty) {
    core->set_property(target_device, properties);

    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(property.empty()) << property_item.first << ":" << property.as<std::string>() << '\n';
    }
}

TEST_P(OVPropertiesIncorrectTestsNPU, SetPropertiesWithIncorrectKey) {
    core->get_versions(target_device);
    std::vector<ov::PropertyName> supported_properties;
    supported_properties = core->get_property(target_device, ov::supported_properties);

    for (const auto& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_FALSE(supported);
    }
}

TEST_P(OVPropertiesEnvVarTestsNPU, WrongEnvVarsDontAffectPluginLoading) {
    const std::string wrong_env_var_value = "WRONG_ENV_VAR_VALUE";
    std::ignore = core->get_available_devices();
    core->unload_plugin(target_device);
    for (const auto& property_item : properties) {
        if (std::getenv(property_item.second.as<std::string>().c_str()) == nullptr) {
            set_env(property_item.second.as<std::string>(), wrong_env_var_value);
            std::vector<std::string> plugins;
            try {
                plugins = core->get_available_devices();
            } catch (const std::exception& e) {
                FAIL() << "Failed to get available devices due to error: " << e.what();
            }
            unset_env(property_item.second.as<std::string>());
            if (!plugins.empty()) {
                auto it = std::find_if(plugins.begin(),
                                       plugins.end(),
                                       [&target_device = std::as_const(target_device)](const std::string& plugin) {
                                           return plugin.find(target_device) != std::string::npos;
                                       });
                ASSERT_TRUE(it != plugins.end()) << "Plugin was not loaded with wrong environment value for "
                                                 << property_item.second.as<std::string>();
            }
        }
    }
}

TEST_P(OVCheckSetSupportedRWMetricsPropsTestsNPU, ChangeCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));

    for (const auto& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_FALSE(supported);

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(default_property.empty());

        if (ov::PropertyName(property_item.first).is_mutable() && !property_item.second.empty()) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, {property_item}));
            core->compile_model(model, target_device, compileModelProperties);
            ov::Any actual_property_value;
            OV_ASSERT_NO_THROW(actual_property_value = core->get_property(target_device, property_item.first));
            ASSERT_FALSE(actual_property_value.empty());

            std::string expect_value = property_item.second.as<std::string>();
            std::string actual_value = actual_property_value.as<std::string>();
            EXPECT_EQ(actual_value, expect_value) << "Property is changed in wrong way";
        }
    }
}

const std::vector<ov::AnyMap> compat_CorrectPluginMutableProperties = {
    {{ov::internal::exclusive_async_requests.name(), true}},
    {{ov::intel_npu::dma_engines.name(), 1}},
    {{ov::intel_npu::compilation_mode.name(), "DefaultHW"}},
    {{ov::intel_npu::profiling_type.name(), ov::intel_npu::ProfilingType::INFER}}};

const std::vector<ov::AnyMap> compat_IncorrectMutablePropertiesWrongValueTypes = {
    {{ov::intel_npu::compilation_mode.name(), -3.6}},
    {{ov::intel_npu::profiling_type.name(), 10}},
    {{ov::intel_npu::dma_engines.name(), false}},
};

const std::vector<ov::AnyMap> CorrectPluginMutableProperties = {
    {{ov::intel_npu::stepping.name(), 0}},
};

const std::vector<ov::AnyMap> IncorrectMutablePropertiesWrongValueTypes = {
    {{ov::intel_npu::stepping.name(), "V1"}},
};

const std::vector<AnyMap> PropsWithEnvVars{
    {{ov::log::level.name(), ::intel_npu::LOG_LEVEL::envVar()}},
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    {{ov::intel_npu::platform.name(), ::intel_npu::PLATFORM::envVar()}},
    {{ov::intel_npu::create_executor.name(), ::intel_npu::CREATE_EXECUTOR::envVar()}},
    {{ov::intel_npu::defer_weights_load.name(), ::intel_npu::DEFER_WEIGHTS_LOAD::envVar()}},
    {{ov::intel_npu::compiler_type.name(), ::intel_npu::COMPILER_TYPE::envVar()}},
    {{ov::intel_npu::compilation_mode.name(), ::intel_npu::COMPILATION_MODE::envVar()}},
    {{ov::intel_npu::dynamic_shape_to_static.name(), ::intel_npu::DYNAMIC_SHAPE_TO_STATIC::envVar()}},
    {{ov::intel_npu::tiles.name(), ::intel_npu::TILES::envVar()}},
    {{ov::intel_npu::dma_engines.name(), ::intel_npu::DMA_ENGINES::envVar()}},
    {{ov::intel_npu::disable_version_check.name(), ::intel_npu::DISABLE_VERSION_CHECK::envVar()}},
    {{ov::intel_npu::export_raw_blob.name(), ::intel_npu::EXPORT_RAW_BLOB::envVar()}},
    {{ov::intel_npu::import_raw_blob.name(), ::intel_npu::IMPORT_RAW_BLOB::envVar()}},
#endif
};

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compat_CorrectPluginMutableProperties)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesTestsNPU>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesIncorrectTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(compat_IncorrectMutablePropertiesWrongValueTypes)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesIncorrectTestsNPU>));

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests,
                         OVPropertiesEnvVarTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(PropsWithEnvVars)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesEnvVarTestsNPU>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCheckSetSupportedRWMetricsPropsTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn([] {
                                                std::vector<ov::AnyMap> combined =
                                                    compat_CorrectPluginMutableProperties;
                                                combined.insert(combined.end(),
                                                                CorrectPluginMutableProperties.begin(),
                                                                CorrectPluginMutableProperties.end());
                                                return combined;
                                            }())),
                         (ov::test::utils::appendPlatformTypeTestName<OVCheckSetSupportedRWMetricsPropsTestsNPU>));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVPropertiesIncorrectTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(IncorrectMutablePropertiesWrongValueTypes)),
                         (ov::test::utils::appendPlatformTypeTestName<OVPropertiesIncorrectTestsNPU>));

}  // namespace ov::test::behavior
