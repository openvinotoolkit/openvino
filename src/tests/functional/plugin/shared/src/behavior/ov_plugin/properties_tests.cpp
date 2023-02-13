// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/properties.hpp"
#include <cstdint>

namespace ov {
namespace test {
namespace behavior {

std::string OVEmptyPropertiesTests::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string target_device = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    return "target_device=" + target_device;
}

void OVEmptyPropertiesTests::SetUp() {
    target_device = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

std::string OVPropertiesTests::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTests::SetUp() {
    std::tie(target_device, properties) = this->GetParam();
    APIBaseTest::SetUp();
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVPropertiesTests::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

std::string OVSetPropComplieModleGetPropTests::getTestCaseName(testing::TestParamInfo<CompileModelPropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    AnyMap compileModelProperties;
    std::tie(target_device, properties, compileModelProperties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    if (!compileModelProperties.empty()) {
        result << "_compileModelProp=" << util::join(util::split(util::to_string(compileModelProperties), ' '), "_");
    }
    return result.str();
}

void OVSetPropComplieModleGetPropTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::tie(target_device, properties, compileModelProperties) = this->GetParam();
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

std::string OVPropertiesTestsWithComplieModelProps::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
    std::string target_device;
    AnyMap properties;
    std::tie(target_device, properties) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    std::ostringstream result;
    result << "target_device=" << target_device << "_";
    if (!properties.empty()) {
        result << "properties=" << util::join(util::split(util::to_string(properties), ' '), "_");
    }
    return result.str();
}

void OVPropertiesTestsWithComplieModelProps::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string temp_device;
    std::tie(temp_device, properties) = this->GetParam();
    APIBaseTest::SetUp();

    std::string::size_type pos = temp_device.find(":", 0);
    std::string hw_device;
    if (pos == std::string::npos) {
        target_device = temp_device;
        hw_device = temp_device;
    } else {
        target_device = temp_device.substr(0, pos);
        hw_device = temp_device.substr(++pos, std::string::npos);
    }

    if (target_device == std::string(CommonTestUtils::DEVICE_MULTI) ||
        target_device == std::string(CommonTestUtils::DEVICE_AUTO) ||
        target_device == std::string(CommonTestUtils::DEVICE_HETERO)) {
        compileModelProperties = { ov::device::priorities(hw_device) };
    } else if (target_device ==  std::string(CommonTestUtils::DEVICE_BATCH)) {
        compileModelProperties = {{ CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG) , hw_device}};
    }
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

void OVPropertiesTestsWithComplieModelProps::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

TEST_P(OVEmptyPropertiesTests, SetEmptyProperties) {
    OV_ASSERT_NO_THROW(core->get_property(target_device, ov::supported_properties));
    OV_ASSERT_NO_THROW(core->set_property(target_device, AnyMap{}));
}

TEST_P(OVPropertiesTests, SetCorrectPropertiesWithoutPlugin) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
}

// Setting correct properties doesn't throw
TEST_P(OVPropertiesTests, SetCorrectProperties) {
    core->get_versions(target_device);
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));
}

TEST_P(OVPropertiesTests, canSetPropertyAndCheckGetProperty) {
    core->set_property(target_device, properties);
    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(property.empty());
        std::cout << property_item.first << ":" << property.as<std::string>() << std::endl;
    }
}

TEST_P(OVGetPropertiesIncorrectTests, GetIncoorectProperties) {
    ov::Core ie = createCoreWithTemplate();
    for (const auto& property_item : properties) {
        ASSERT_THROW(ie.get_property(target_device, property_item.first), ov::Exception);
    }
}

TEST_P(OVPropertiesIncorrectTests, SetPropertiesForUnregisteredDevice) {
    ASSERT_THROW(core->set_property("unregistered_device", properties), ov::Exception);
}

TEST_P(OVPropertiesIncorrectTests, SetPropertiesIncorrectProperies) {
    core->get_versions(target_device);
    ASSERT_THROW(core->set_property(target_device, properties), ov::Exception);
}

TEST_P(OVPropertiesIncorrectTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVClassGetConfigTest, GetConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<ov::PropertyName> configValues;

    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto&& confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVPropertiesDefaultTests, CanSetDefaultValueBackToPlugin) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (auto& supported_property : supported_properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, supported_property));
        if (supported_property.is_mutable()) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, {{ supported_property, property}}));
        }
    }
}

TEST_P(OVPropertiesDefaultTests, CheckDefaultValues) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (auto&& default_property : properties) {
        auto supported = util::contains(supported_properties, default_property.first);
        ASSERT_TRUE(supported) << "default_property=" << default_property.first;
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, default_property.first));
        ASSERT_EQ(default_property.second, property);
    }
}

TEST_P(OVSetPropComplieModleGetPropTests, SetPropertyComplieModelGetProperty) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));

    ov::CompiledModel exeNetWork;
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));
    if (target_device == CommonTestUtils::DEVICE_AUTO)
        GTEST_SKIP();

    for (const auto& property_item : compileModelProperties) {
        Any exeNetProperty;
        OV_ASSERT_NO_THROW(exeNetProperty = exeNetWork.get_property(property_item.first));
        ASSERT_EQ(property_item.second.as<std::string>(), exeNetProperty.as<std::string>());
    }

    //the value of get property should be the same as set property
    for (const auto& property_item : properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core->get_property(target_device, property_item.first));
        ASSERT_EQ(property_item.second.as<std::string>(), property.as<std::string>());
    }
}

TEST_P(OVSetPropComplieModleWihtIncorrectPropTests, CanNotCompileModelWithIncorrectProperties) {
    ASSERT_THROW(core->compile_model(model, target_device, properties), ov::Exception);
}

TEST_P(OVSetSupportPropComplieModleWithoutConfigTests, SetPropertyCompiledModelWithCorrectProperty) {
    ASSERT_NO_THROW(core->compile_model(model, target_device, properties));
}

std::string OVCompileModelGetExecutionDeviceTests::getTestCaseName(testing::TestParamInfo<OvPropertiesParams> obj) {
    std::string target_device;
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(target_device, userConfig) = obj.param;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    auto compileModelProperties = userConfig.first;
    std::ostringstream result;
    result << "device_name=" << target_device << "_";
    if (!compileModelProperties.empty()) {
        result << "_compileModelProp=" << util::join(util::split(util::to_string(compileModelProperties), ' '), "_");
    }
    result << "_expectedDevice=" << userConfig.second;
    return result.str();
}

void OVCompileModelGetExecutionDeviceTests::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::pair<ov::AnyMap, std::string> userConfig;
    std::tie(target_device, userConfig) = GetParam();
    compileModelProperties = userConfig.first;
    expectedDeviceName = userConfig.second;
    model = ngraph::builder::subgraph::makeConvPoolRelu();
}

TEST_P(OVCompileModelGetExecutionDeviceTests, CanGetExecutionDeviceInfo) {
    ov::CompiledModel exeNetWork;
    auto deviceList = core->get_available_devices();
    std::vector<std::string> expected_devices = util::split(expectedDeviceName, ',');
    std::vector<std::string> updatedExpectDevices;
    updatedExpectDevices.assign(expected_devices.begin(), expected_devices.end());
    for (auto &iter : compileModelProperties) {
        if ((iter.first == ov::hint::performance_mode && iter.second.as<ov::hint::PerformanceMode>() == ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT) ||
            target_device.find("MULTI") != std::string::npos) {
            for (auto& deviceName : expected_devices) {
                for (auto&& device : deviceList) {
                    if (device.find(deviceName) != std::string::npos) {
                        auto updatedExpectDevices_iter = std::find(updatedExpectDevices.begin(), updatedExpectDevices.end(), deviceName);
                        if (updatedExpectDevices_iter != updatedExpectDevices.end())
                            updatedExpectDevices.erase(updatedExpectDevices_iter);
                        updatedExpectDevices.push_back(std::move(device));
                    }
                }
            }
            break;
        }
    }
    std::sort(updatedExpectDevices.begin(), updatedExpectDevices.end());
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));
    ov::Any property;
    OV_ASSERT_NO_THROW(property = exeNetWork.get_property(ov::execution_devices));
    std::vector<std::string> property_vector = property.as<std::vector<std::string>>();
    std::sort(property_vector.begin(), property_vector.end());
    if (expectedDeviceName.find("undefined") == std::string::npos)
        ASSERT_EQ(property_vector, updatedExpectDevices);
    else
        ASSERT_FALSE(property.empty());
}

TEST_P(OVClassExecutableNetworkGetMetricTest_EXEC_DEVICES, CanGetExecutionDeviceInfo) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> expectedTargets = {expectedDeviceName};
    auto compiled_model = ie.compile_model(model, target_device, compileModelProperties);

    std::vector<std::string> exeTargets;
    OV_ASSERT_NO_THROW(exeTargets = compiled_model.get_property(ov::execution_devices));

    ASSERT_EQ(expectedTargets, exeTargets);
}

std::vector<ov::AnyMap> OVPropertiesTestsWithComplieModelProps::getAllROPropertiesValues() {
    std::vector<ov::AnyMap> res;
    // Read Only
    res.push_back({{ov::PropertyName(ov::available_devices.name(), ov::available_devices.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::model_name.name(), ov::model_name.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::optimal_number_of_infer_requests.name(), ov::optimal_number_of_infer_requests.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::loaded_from_cache.name(), ov::loaded_from_cache.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::range_for_streams.name(), ov::range_for_streams.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::range_for_async_infer_requests.name(), ov::range_for_async_infer_requests.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::full_name.name(), ov::device::full_name.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::architecture.name(), ov::device::architecture.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::uuid.name(), ov::device::uuid.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::type.name(), ov::device::type.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::gops.name(), ov::device::gops.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::thermal.name(), ov::device::thermal.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::capabilities.name(), ov::device::capabilities.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::execution_devices.name(), ov::execution_devices.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::priorities.name(), ov::device::priorities.mutability), nullptr}});

    return res;
}

std::vector<ov::AnyMap> OVPropertiesTestsWithComplieModelProps::getRWPropertiesValues(std::vector<std::string> param_properties) {
    std::vector<ov::AnyMap> res;

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::hint::inference_precision) != param_properties.end()) {
        // Writable
        const std::vector<ov::element::Type> ovElemTypes = {
            ov::element::f64, ov::element::f32, ov::element::f16,
            ov::element::i64, ov::element::i32, ov::element::i16, ov::element::i8,
            ov::element::u64, ov::element::u32, ov::element::u16, ov::element::u8,
            ov::element::boolean,
        };
        for (auto &precision : ovElemTypes) {
            res.push_back({{ov::PropertyName(ov::hint::inference_precision.name(), ov::hint::inference_precision.mutability), precision}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::hint::model_priority) != param_properties.end()) {
        ov::hint::Priority priorities[] = {ov::hint::Priority::LOW , ov::hint::Priority::MEDIUM, ov::hint::Priority::HIGH};
        for (auto &priority : priorities) {
            res.push_back({{ov::PropertyName(ov::hint::model_priority.name(), ov::hint::model_priority.mutability), priority}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::hint::performance_mode) != param_properties.end()) {
        ov::hint::PerformanceMode performance_modes[] = {ov::hint::PerformanceMode::UNDEFINED , ov::hint::PerformanceMode::LATENCY,
                ov::hint::PerformanceMode::THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT};
        for (auto &performance_mode : performance_modes) {
            res.push_back({{ov::PropertyName(ov::hint::performance_mode.name(), ov::hint::performance_mode.mutability), performance_mode}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::hint::execution_mode) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::hint::num_requests.name(), ov::hint::num_requests.mutability), 1}});

        res.push_back({{ov::PropertyName(ov::hint::allow_auto_batching.name(), ov::hint::allow_auto_batching.mutability), true}});
        res.push_back({{ov::PropertyName(ov::hint::allow_auto_batching.name(), ov::hint::allow_auto_batching.mutability), false}});

        ov::hint::ExecutionMode execution_modes[] = {ov::hint::ExecutionMode::UNDEFINED , ov::hint::ExecutionMode::PERFORMANCE,
                                                    ov::hint::ExecutionMode::ACCURACY};
        for (auto &execution_mode : execution_modes) {
            res.push_back({{ov::PropertyName(ov::hint::execution_mode.name(), ov::hint::execution_mode.mutability), execution_mode}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::log::level) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::enable_profiling.name(), ov::enable_profiling.mutability), true}});
        res.push_back({{ov::PropertyName(ov::enable_profiling.name(), ov::enable_profiling.mutability), false}});

        ov::log::Level log_levels[] = {ov::log::Level::NO , ov::log::Level::ERR, ov::log::Level::WARNING,
                                    ov::log::Level::INFO, ov::log::Level::DEBUG, ov::log::Level::TRACE};
        for (auto &log_level : log_levels) {
            res.push_back({{ov::PropertyName(ov::log::level.name(), ov::log::level.mutability), log_level}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::cache_dir) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::cache_dir.name(), ov::cache_dir.mutability), ov::cache_dir("./tmp_cache_dir")}});
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::auto_batch_timeout) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::auto_batch_timeout.name(), ov::auto_batch_timeout.mutability), 2147483647}});
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::force_tbb_terminate) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::force_tbb_terminate.name(), ov::force_tbb_terminate.mutability), true}});
        res.push_back({{ov::PropertyName(ov::force_tbb_terminate.name(), ov::force_tbb_terminate.mutability), false}});
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::streams::num) != param_properties.end()) {
        ov::streams::Num nums[] = {ov::streams::AUTO, ov::streams::NUMA};
        for (auto &num : nums) {
            res.push_back({{ov::PropertyName(ov::streams::num.name(), ov::streams::num.mutability), num}});
            // res.push_back({{ov::PropertyName(ov::num_streams.name(), ov::num_streams.mutability), num}});
        }
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::inference_num_threads) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::inference_num_threads.name(), ov::inference_num_threads.mutability), 1}});
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::compilation_num_threads) != param_properties.end()) {
        res.push_back({{ov::PropertyName(ov::compilation_num_threads.name(), ov::compilation_num_threads.mutability), 1}});
    }

    if (param_properties.empty() || std::find(param_properties.begin(), param_properties.end(), ov::affinity) != param_properties.end()) {
        ov::Affinity affinities[] = {ov::Affinity::NONE , ov::Affinity::CORE, ov::Affinity::NUMA, ov::Affinity::HYBRID_AWARE};
        for (auto &affinity : affinities) {
            res.push_back({{ov::PropertyName(ov::affinity.name(), ov::affinity.mutability), affinity}});
        }
    }

    return res;
}

TEST_P(OVCheckChangePropComplieModleGetPropTestsRW, GetCheckSupportedAndChangeCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));

        if (property_item.first.is_mutable()) {
            if (property_item.second.empty()) {
                continue;
            }
            OV_ASSERT_NO_THROW(core->set_property(target_device, {property_item}));
            core->compile_model(model, target_device, compileModelProperties);
            ov::Any new_property_value;
            OV_ASSERT_NO_THROW(new_property_value = core->get_property(target_device, property_item.first));
            if (default_property != property_item.second) {
                ASSERT_TRUE(new_property_value == property_item.second) << "Peoperty is not changed";
            } else {
                ASSERT_TRUE(new_property_value == property_item.second) << "Peoperty is changed in wrong way";
            }
        } else {
            FAIL() << "Property is not mutable";
        }
    }
}

TEST_P(OVCheckChangePropComplieModleGetPropTestsRO, GetAndCheckSupportedCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_value;
        OV_ASSERT_NO_THROW(default_value = core->get_property(target_device, property_item.first));

        ASSERT_TRUE(default_value.empty()) << "Peoperty value is empty";
    }
}

TEST_P(OVCheckChangePropComplieModleGetPropTests_DEVICE_ID, ChangeCorrectDeviceProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));

    std::vector<std::string> availableDevices = core->get_available_devices();
    for (int id = 0; id < availableDevices.size(); id++) {
        std::string deviceName = availableDevices[id];
        auto supported = util::contains(supported_properties, ov::device::id);
        ASSERT_TRUE(supported) << "property is not supported: " << ov::device::id;

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, ov::device::id));

        OV_ASSERT_NO_THROW(core->set_property(target_device, {ov::device::id(id)}));
        core->compile_model(model, target_device, compileModelProperties);
        ov::Any new_property_value;
        OV_ASSERT_NO_THROW(new_property_value = core->get_property(target_device, ov::device::id));
        if (default_property != deviceName) {
            ASSERT_TRUE(new_property_value == deviceName) << "Peoperty is not changed";
        } else {
            ASSERT_TRUE(new_property_value == deviceName) << "Peoperty is changed in wrong way";
        }
    }
}

std::vector<ov::AnyMap> OVPropertiesTestsWithComplieModelProps::getModelDependcePropertiesValues() {
    std::vector<ov::AnyMap> res;
    // Read Only
    res.push_back({{ov::PropertyName(ov::optimal_batch_size.name(), ov::optimal_batch_size.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::max_batch_size.name(), ov::max_batch_size.mutability), nullptr}});
    return res;
}

TEST_P(OVCheckChangePropComplieModleGetPropTests_ModelDependceProps, ChangeCorrectDeviceProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    auto supported = util::contains(supported_properties, ov::hint::model);
    ASSERT_TRUE(supported) << "property is not supported: " << ov::hint::model;

    OV_ASSERT_NO_THROW(core->set_property(target_device, ov::hint::model(model)));
    core->compile_model(model, target_device, compileModelProperties);

    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
    }
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start{0}, end{0}, step{0};

    ASSERT_NO_THROW(std::tie(start, end, step) = ie.get_property(target_device, ov::range_for_async_infer_requests));

    std::cout << "Range for async infer requests: " << std::endl
    << start << std::endl
    << end << std::endl
    << step << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetMetricTest_RANGE_FOR_STREAMS, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    unsigned int start = 0, end = 0;

    ASSERT_NO_THROW(std::tie(start, end) = ie.get_property(target_device, ov::range_for_streams));

    std::cout << "Range for streams: " << std::endl
    << start << std::endl
    << end << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

//
// DeviceID Property
//

TEST_P(OVClassGetMetricTest_FULL_DEVICE_NAME_with_DEVICE_ID, GetMetricAndPrintNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto device_ids = ie.get_property(target_device, ov::available_devices);
        ASSERT_GT(device_ids.size(), 0);
        std::string t;
        OV_ASSERT_NO_THROW(t = ie.get_property(target_device, ov::device::full_name, ov::device::id(device_ids.front())));
        std::cout << "Device " << device_ids.front() << " " <<  ", Full device name: " << std::endl << t << std::endl;
        OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
    } else {
        GTEST_FAIL() << "Device id is not supported";
    }
}

TEST_P(OVClassSpecificDeviceTestGetConfig, GetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    ov::Any p;

    std::string deviceID, clear_target_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    if (!supportsDeviceID(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
    if (!supportsAvailableDevices(ie, clear_target_device)) {
        GTEST_FAIL() << "Device does not support AvailableDevices property" << std::endl;
    }
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL() << "No DeviceID" << std::endl;
    }

    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto &&confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVClassGetAvailableDevices, GetAvailableDevicesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    std::vector<std::string> devices;

    OV_ASSERT_NO_THROW(devices = ie.get_available_devices());

    bool deviceFound = false;
    std::cout << "Available devices: " << std::endl;
    for (auto&& device : devices) {
        if (device.find(target_device) != std::string::npos) {
            deviceFound = true;
        }

        std::cout << device << " ";
    }
    std::cout << std::endl;

    ASSERT_TRUE(deviceFound);
}

TEST_P(OVClassSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    std::string deviceID, cleartarget_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    if (!supportsDeviceID(ie, cleartarget_device) || !supportsAvailableDevices(ie, cleartarget_device)) {
        GTEST_FAIL();
    }
    auto deviceIDs = ie.get_property(cleartarget_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL();
    }

    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
    bool value = false;
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::enable_profiling));
    ASSERT_TRUE(value);
}

TEST_P(OVClassSetDefaultDeviceIDTest, SetDefaultDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL();
    }
    std::string value;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::device::id(deviceID), ov::enable_profiling(true)));
    ASSERT_TRUE(ie.get_property(target_device, ov::enable_profiling));
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::enable_profiling.name()).as<std::string>());
    ASSERT_EQ(value, "YES");
}

TEST_P(OVClassSetGlobalConfigTest, SetGlobalConfigNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    auto deviceIDs = ie.get_property(target_device, ov::available_devices);
    ov::Any ref, src;
    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(ie.set_property(target_device + "." + dev_id, ov::enable_profiling(false)));
    }
    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
    OV_ASSERT_NO_THROW(ref = ie.get_property(target_device, ov::enable_profiling.name()));

    for (auto& dev_id : deviceIDs) {
        OV_ASSERT_NO_THROW(src = ie.get_property(target_device + "." + dev_id, ov::enable_profiling.name()));
        ASSERT_EQ(src, ref);
    }
}
}  // namespace behavior
}  // namespace test
}  // namespace ov
