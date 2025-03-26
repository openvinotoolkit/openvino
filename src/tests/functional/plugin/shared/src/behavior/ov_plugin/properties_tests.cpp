// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>

#include "behavior/ov_plugin/properties_tests.hpp"
#include "openvino/runtime/properties.hpp"
#include "common_test_utils/subgraph_builders/split_concat.hpp"

namespace ov {
namespace test {
namespace behavior {

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
    model = ov::test::utils::make_split_concat();
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
    model = ov::test::utils::make_split_concat();
}

std::string OVPropertiesTestsWithCompileModelProps::getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
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

void OVPropertiesTestsWithCompileModelProps::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    std::string temp_device;
    std::tie(temp_device, properties) = this->GetParam();

    std::string::size_type pos = temp_device.find(":", 0);
    if (pos != std::string::npos) {
        target_device = temp_device.substr(0, pos);
        for (auto& it : compileModelProperties) {
            OPENVINO_ASSERT(it.first == ov::device::priorities.name(),
                            "there is already ov::device::priorities() in compileModelProperties");
        }
        compileModelProperties.insert(ov::device::priorities(temp_device.substr(++pos, std::string::npos)));
    } else {
        target_device = temp_device;
    }

    model = ov::test::utils::make_split_concat();

    APIBaseTest::SetUp();
}

void OVPropertiesTestsWithCompileModelProps::TearDown() {
    if (!properties.empty()) {
        utils::PluginCache::get().reset();
    }
    APIBaseTest::TearDown();
}

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

TEST_P(OVPropertiesIncorrectTests, SetPropertiesWithIncorrectKey) {
    core->get_versions(target_device);
    ASSERT_THROW(core->set_property(target_device, properties), ov::Exception);
}

TEST_P(OVBasicPropertiesTestsP, GetMetricThrowUnsupported) {
    ov::Core ie = ov::test::utils::create_core();
    ASSERT_THROW(ie.get_property(target_device, "unsupported_metric"), ov::Exception);
}

TEST_P(OVBasicPropertiesTestsP, SetConfigAllThrows) {
    ov::Core core;
#if !defined(OPENVINO_STATIC_LIBRARY) && !defined(USE_STATIC_IE)
    ov::test::utils::register_template_plugin(core);
#endif  // !OPENVINO_STATIC_LIBRARY && !USE_STATIC_IE
    OV_ASSERT_NO_THROW(core.set_property({{"unsupported_key", "4"}}));
    ASSERT_ANY_THROW(core.get_versions(target_device));
}

TEST_P(OVBasicPropertiesTestsP, SetConfigForUnRegisteredDeviceThrows) {
    ov::Core ie = ov::test::utils::create_core();
    ASSERT_THROW(ie.set_property("unregistered_device", {{"unsupported_key", "4"}}), ov::Exception);
}

TEST_P(OVBasicPropertiesTestsP, getVersionsByDeviceClassNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    OV_ASSERT_NO_THROW(ie.get_versions(target_device));
}

TEST_P(OVBasicPropertiesTestsP, getVersionsByExactDeviceNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    std::vector<std::string> devices;
    devices = ie.get_available_devices();

    std::cout << "Available devices: " << std::endl;
    for (auto&& device : devices) {
        OV_ASSERT_NO_THROW(ie.get_versions(device));
    }
}

TEST_P(OVPropertiesDefaultSupportedTests, CanSetDefaultValueBackToPlugin) {
    ov::Core core = ov::test::utils::create_core();
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core.get_property(target_device, ov::supported_properties));
    for (auto& supported_property : supported_properties) {
        Any property;
        OV_ASSERT_NO_THROW(property = core.get_property(target_device, supported_property));
        ASSERT_FALSE(property.empty());
        if (supported_property.is_mutable()) {
            OV_ASSERT_NO_THROW(core.set_property(target_device, {{ supported_property, property}}));
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

TEST_P(OVSetPropComplieModleGetPropTests, SetPropertyAndComplieModelWithPropsWorkCorrectTogeter) {
    OV_ASSERT_NO_THROW(core->set_property(target_device, properties));

    ov::CompiledModel exeNetWork;
    OV_ASSERT_NO_THROW(exeNetWork = core->compile_model(model, target_device, compileModelProperties));

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

std::vector<ov::AnyMap> OVPropertiesTestsWithCompileModelProps::getROMandatoryProperties(bool is_sw_device) {
    std::vector<ov::AnyMap> res;
    res.push_back({{ov::PropertyName(ov::device::full_name.name(), ov::device::full_name.mutability), nullptr}});
    if (!is_sw_device) {
        res.push_back({{ov::PropertyName(ov::device::architecture.name(), ov::device::architecture.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::device::type.name(), ov::device::type.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::execution_devices.name(), ov::execution_devices.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::available_devices.name(), ov::available_devices.mutability), nullptr}});
        res.push_back(
            {{ov::PropertyName(ov::hint::execution_mode.name(), ov::hint::execution_mode.mutability), nullptr}});
        res.push_back(
            {{ov::PropertyName(ov::hint::inference_precision.name(), ov::hint::inference_precision.mutability),
              nullptr}});
    }

    return res;
}

std::vector<ov::AnyMap> OVPropertiesTestsWithCompileModelProps::getROOptionalProperties(bool is_sw_device) {
    std::vector<ov::AnyMap> res;
    if (is_sw_device) {
        res.push_back({{ov::PropertyName(ov::device::architecture.name(), ov::device::architecture.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::device::type.name(), ov::device::type.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::execution_devices.name(), ov::execution_devices.mutability), nullptr}});
        res.push_back({{ov::PropertyName(ov::available_devices.name(), ov::available_devices.mutability), nullptr}});
        res.push_back(
            {{ov::PropertyName(ov::hint::execution_mode.name(), ov::hint::execution_mode.mutability), nullptr}});
        res.push_back(
            {{ov::PropertyName(ov::hint::inference_precision.name(), ov::hint::inference_precision.mutability),
              nullptr}});
    }
    res.push_back({{ov::PropertyName(ov::loaded_from_cache.name(), ov::loaded_from_cache.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::uuid.name(), ov::device::uuid.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::luid.name(), ov::device::luid.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::gops.name(), ov::device::gops.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::device::thermal.name(), ov::device::thermal.mutability), nullptr}});

    return res;
}

std::vector<ov::AnyMap> OVPropertiesTestsWithCompileModelProps::configureProperties(std::vector<std::string> props) {
    std::vector<ov::AnyMap> res;

    for (auto &prop : props) {
        res.push_back({{ov::PropertyName(prop, ov::PropertyMutability::RO), nullptr}});
    }

    return res;
}

std::vector<ov::AnyMap>
OVPropertiesTestsWithCompileModelProps::getRWMandatoryPropertiesValues(
    const std::vector<std::string>& props, bool is_sw_device) {
    std::vector<ov::AnyMap> res;

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::performance_mode.name()) != props.end()) {
        ov::hint::PerformanceMode performance_modes[] = {ov::hint::PerformanceMode::LATENCY,
                ov::hint::PerformanceMode::THROUGHPUT, ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT};
        for (auto &performance_mode : performance_modes) {
            res.push_back({{ov::hint::performance_mode(performance_mode)}});
        }
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::num_requests.name()) != props.end()) {
        res.push_back({{ov::hint::num_requests(1)}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::execution_mode.name()) != props.end()) {
        ov::hint::ExecutionMode execution_modes[] = {ov::hint::ExecutionMode::PERFORMANCE, ov::hint::ExecutionMode::ACCURACY};
        for (auto &execution_mode : execution_modes) {
            res.push_back({{ov::hint::execution_mode(execution_mode)}});
        }
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::enable_profiling.name()) != props.end()) {
        res.push_back({{ov::enable_profiling(true)}});
        res.push_back({{ov::enable_profiling(false)}});
    }

    if (!is_sw_device) {
        if (props.empty() || std::find(props.begin(), props.end(), ov::streams::num.name()) != props.end()) {
            res.push_back({ov::streams::num(3)});
        }
    }

    return res;
}

std::vector<ov::AnyMap>
OVPropertiesTestsWithCompileModelProps::getWrongRWMandatoryPropertiesValues(
    const std::vector<std::string>& props, bool is_sw_device) {
    std::vector<ov::AnyMap> res;

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::performance_mode.name()) != props.end()) {
        res.push_back({{ov::hint::performance_mode.name(), -1}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::num_requests.name()) != props.end()) {
        res.push_back({{ov::hint::num_requests.name(), -10}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::execution_mode.name()) != props.end()) {
        res.push_back({{ov::hint::execution_mode.name(), 5}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::enable_profiling.name()) != props.end()) {
        res.push_back({{ov::enable_profiling.name(), -1}});
    }

    if (!is_sw_device) {
        if (props.empty() || std::find(props.begin(), props.end(), ov::streams::num.name()) != props.end()) {
            res.push_back({ov::streams::num(-10)});
        }
    }

    return res;
}

std::vector<ov::AnyMap>
OVPropertiesTestsWithCompileModelProps::getRWOptionalPropertiesValues(
    const std::vector<std::string>& props, bool is_sw_device) {
    std::vector<ov::AnyMap> res;

    if (props.empty() || std::find(props.begin(), props.end(), ov::inference_num_threads.name()) != props.end()) {
        res.push_back({ov::inference_num_threads(1)});
        res.push_back({ov::compilation_num_threads(1)});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::enable_hyper_threading.name()) != props.end()) {
        res.push_back({ov::hint::enable_hyper_threading(true)});
        res.push_back({ov::hint::enable_hyper_threading(false)});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::enable_cpu_pinning.name()) != props.end()) {
        res.push_back({ov::hint::enable_cpu_pinning(true)});
        res.push_back({ov::hint::enable_cpu_pinning(false)});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::scheduling_core_type.name()) != props.end()) {
        ov::hint::SchedulingCoreType schedulingCoreTypes[] = {ov::hint::SchedulingCoreType::ANY_CORE,
                                                              ov::hint::SchedulingCoreType::PCORE_ONLY,
                                                              ov::hint::SchedulingCoreType::ECORE_ONLY};
        for (auto &schedulingCoreType : schedulingCoreTypes) {
            res.push_back({ov::hint::scheduling_core_type(schedulingCoreType)});
        }
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::enable_mmap.name()) != props.end()) {
        res.push_back({ov::enable_mmap(true)});
        res.push_back({ov::enable_mmap(false)});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::log::level.name()) != props.end()) {
        ov::log::Level log_levels[] = {ov::log::Level::NO , ov::log::Level::ERR, ov::log::Level::WARNING,
                                       ov::log::Level::INFO, ov::log::Level::DEBUG, ov::log::Level::TRACE};
        for (auto &log_level : log_levels) {
            res.push_back({ov::log::level(log_level)});
        }
    }

    if (is_sw_device) {
        if (props.empty() || std::find(props.begin(), props.end(), ov::streams::num.name()) != props.end()) {
            res.push_back({ov::streams::num(3)});
        }
    }

    return res;
}

std::vector<ov::AnyMap>
OVPropertiesTestsWithCompileModelProps::getWrongRWOptionalPropertiesValues(
    const std::vector<std::string>& props, bool is_sw_device) {
    std::vector<ov::AnyMap> res;

    if (props.empty() || std::find(props.begin(), props.end(), ov::inference_num_threads.name()) != props.end()) {
        res.push_back({{ov::inference_num_threads.name(), -1}});
        res.push_back({{ov::compilation_num_threads.name(), -1}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::enable_hyper_threading.name()) != props.end()) {
        res.push_back({{ov::hint::enable_hyper_threading.name(), -1}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::enable_cpu_pinning.name()) != props.end()) {
        res.push_back({{ov::hint::enable_cpu_pinning.name(), -1}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::hint::scheduling_core_type.name()) != props.end()) {
        res.push_back({{ov::hint::scheduling_core_type.name(), -1}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::enable_mmap.name()) != props.end()) {
        res.push_back({{ov::enable_mmap.name(), -10}});
    }

    if (props.empty() || std::find(props.begin(), props.end(), ov::log::level.name()) != props.end()) {
        res.push_back({{ov::log::level.name(), -3}});
    }

    if (is_sw_device) {
        if (props.empty() || std::find(props.begin(), props.end(), ov::streams::num.name()) != props.end()) {
            res.push_back({ov::streams::num(-10)});
        }
    }

    return res;
}

TEST_P(OVCheckSetIncorrectRWMetricsPropsTests, ChangeIncorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        EXPECT_THROW(core->set_property(target_device, {property_item}), ov::Exception);

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(default_property.empty());
        core->compile_model(model, target_device, compileModelProperties);
    }
}

TEST_P(OVCheckSetSupportedRWMetricsPropsTests, ChangeCorrectProperties) {
    std::vector<ov::PropertyName>supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(default_property.empty());

        if (property_item.first.is_mutable() && !property_item.second.empty()) {
            OV_ASSERT_NO_THROW(core->set_property(target_device, {property_item}));
            core->compile_model(model, target_device, compileModelProperties);
            ov::Any actual_property_value;
            OV_ASSERT_NO_THROW(actual_property_value = core->get_property(target_device, property_item.first));
            ASSERT_FALSE(actual_property_value.empty());

            std::string expect_value = property_item.second.as<std::string>();
            std::string actual_value = actual_property_value.as<std::string>();
            EXPECT_EQ(actual_value, expect_value) << "Property changed incorrectly";
        }
    }
}

TEST_P(OVCheckGetSupportedROMetricsPropsTests, ChangeCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    for (const std::pair<ov::PropertyName, ov::Any>& property_item : properties) {
        auto supported = util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_property;
        OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, property_item.first));
        ASSERT_FALSE(default_property.empty());
    }
}

TEST_P(OVCheckChangePropComplieModleGetPropTests_DEVICE_ID, ChangeCorrectDeviceProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    auto supported = util::contains(supported_properties, ov::device::id);
    ASSERT_TRUE(supported) << "property is not supported: " << ov::device::id;

    auto device_ids = core->get_available_devices();
    for (auto&& device_name_with_id : device_ids) {
        if (device_name_with_id.find(target_device) == std::string::npos) {
            continue;
        }

        std::string device_name = device_name_with_id;
        std::string device_id = "";
        auto pos = device_name_with_id.find('.');
        if (pos != std::string::npos) {
            device_name = device_name_with_id.substr(0, pos);
            device_id =  device_name_with_id.substr(pos + 1,  device_name_with_id.size());
        }

        std::string full_name;
        OV_ASSERT_NO_THROW(full_name = core->get_property(device_name, ov::device::full_name, ov::device::id(device_id)));
        ASSERT_FALSE(full_name.empty());

        if (device_id != "") {
            OV_ASSERT_NO_THROW(core->set_property(device_name, {ov::device::id(device_id)}));
            core->compile_model(model, device_name, compileModelProperties);
            std::string actual_device_id;
            OV_ASSERT_NO_THROW(actual_device_id = core->get_property(device_name, ov::device::id));
            EXPECT_EQ(device_id, actual_device_id) << "DeviceID is changed, but new value is not correct";
        }
    }
}

TEST_P(OVCheckChangePropComplieModleGetPropTests_InferencePrecision, ChangeCorrectProperties) {
    std::vector<ov::PropertyName> supported_properties;
    OV_ASSERT_NO_THROW(supported_properties = core->get_property(target_device, ov::supported_properties));
    auto supported = util::contains(supported_properties, ov::hint::inference_precision);
    ASSERT_TRUE(supported) << "property is not supported: " << ov::hint::inference_precision;

    ov::Any default_property;
    OV_ASSERT_NO_THROW(default_property = core->get_property(target_device, ov::hint::inference_precision));
    ASSERT_FALSE(default_property.empty());

    const std::vector<ov::element::Type> ovElemTypes = {ov::element::f64,
                                                        ov::element::f32,
                                                        ov::element::f16,
                                                        ov::element::bf16,
                                                        ov::element::i64,
                                                        ov::element::i32,
                                                        ov::element::i16,
                                                        ov::element::i8,
                                                        ov::element::i4,
                                                        ov::element::u64,
                                                        ov::element::u32,
                                                        ov::element::u16,
                                                        ov::element::u8,
                                                        ov::element::u4,
                                                        ov::element::u1,
                                                        ov::element::boolean,
                                                        ov::element::dynamic};

    bool any_supported = false;
    for (ov::element::Type type : ovElemTypes) {
        try {
            core->set_property(target_device, ov::hint::inference_precision(type));
            core->compile_model(model, target_device, compileModelProperties);
        } catch (const Exception& ex) {
            std::string err_msg(ex.what());
            ASSERT_TRUE(err_msg.find("Wrong value") != std::string::npos ||
                        err_msg.find("Unsupported precision") != std::string::npos) <<
                        "Error message is unclear. The err msg:" << err_msg << std::endl;
            ASSERT_TRUE(err_msg.find("Supported values") != std::string::npos) <<
                        "The error message doesn't provide info about supported precicions." <<
                        "The err msg: " << err_msg << std::endl;
            continue;
        }

        ov::Any actual_property_value;
        OV_ASSERT_NO_THROW(actual_property_value = core->get_property(target_device, ov::hint::inference_precision));
        ASSERT_FALSE(actual_property_value.empty());

        ov::element::Type actual_value = actual_property_value.as<ov::element::Type>();
        ASSERT_EQ(actual_value, type) << "Property changed incorrectly";

        std::cout << "Supported precision: " << type << std::endl;
        any_supported = true;
    }
    ASSERT_TRUE(any_supported) << "No one supported precision is found";
}

std::vector<ov::AnyMap> OVPropertiesTestsWithCompileModelProps::getModelDependcePropertiesValues() {
    std::vector<ov::AnyMap> res;
    // Read Only
    res.push_back({{ov::PropertyName(ov::optimal_batch_size.name(), ov::optimal_batch_size.mutability), nullptr}});
    res.push_back({{ov::PropertyName(ov::max_batch_size.name(), ov::max_batch_size.mutability), nullptr}});
    return res;
}

TEST_P(OVCheckMetricsPropsTests_ModelDependceProps, ChangeCorrectDeviceProperties) {
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
        ASSERT_FALSE(default_property.empty());
    }
}

TEST_P(OVClassSetDefaultDeviceIDPropTest, SetDefaultDeviceIDNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    // sw plugins are not requested to support `ov::available_devices` and ` ov::device::id` property
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

TEST_P(OVSpecificDeviceSetConfigTest, GetConfigSpecificDeviceNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    ov::Any p;

    std::string deviceID, clear_target_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    // sw plugins are not requested to support `ov::available_devices`, `ov::device::id` and `ov::num_streams` property
    auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
        GTEST_FAIL() << "No DeviceID" << std::endl;
    }
    auto device_name = [&clear_target_device] (const std::string& id) {
        return clear_target_device + "." + id;
    };
    // Set default device id to target device
    OV_ASSERT_NO_THROW(ie.set_property(clear_target_device, ov::device::id(deviceID)));
    // Set num_streams to AUTO for clear device name
    OV_ASSERT_NO_THROW(ie.set_property(clear_target_device, ov::num_streams(ov::streams::AUTO)));
    ASSERT_EQ(ie.get_property(clear_target_device, ov::num_streams), ov::streams::AUTO);
    // Check if it is applied for all devices
    for (auto& id : deviceIDs) {
        ASSERT_EQ(ie.get_property(device_name(id), ov::num_streams),  ov::streams::AUTO);
    }
    for (auto& id : deviceIDs) {
        // Set different properties for different devices
        OV_ASSERT_NO_THROW(ie.set_property(device_name(id), ov::num_streams(std::stoi(id))));
        ASSERT_EQ(ie.get_property(device_name(id), ov::num_streams), std::stoi(id));
    }
    // Check if default device id is still the same
    ASSERT_EQ(ie.get_property(clear_target_device, ov::device::id), deviceID);
    // Check if default property is still equal property to default device
    ASSERT_EQ(ie.get_property(clear_target_device, ov::num_streams), std::stoi(deviceID));
}

TEST_P(OVSpecificDeviceGetConfigTest, GetConfigSpecificDeviceNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    ov::Any p;

    std::string deviceID, clear_target_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    // sw plugins are not requested to support `ov::available_devices` property
    if (!sw_plugin_in_target_device(target_device)) {
        auto deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
        if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
            GTEST_FAIL() << "No DeviceID" << std::endl;
        }
    }

    std::vector<ov::PropertyName> configValues;
    OV_ASSERT_NO_THROW(configValues = ie.get_property(target_device, ov::supported_properties));

    for (auto &&confKey : configValues) {
        ov::Any defaultValue;
        OV_ASSERT_NO_THROW(defaultValue = ie.get_property(target_device, confKey));
        ASSERT_FALSE(defaultValue.empty());
    }
}

TEST_P(OVGetAvailableDevicesPropsTest, GetAvailableDevicesNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
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

TEST_P(OVSpecificDeviceTestSetConfig, SetConfigSpecificDeviceNoThrow) {
    ov::Core ie = ov::test::utils::create_core();

    std::string deviceID, cleartarget_device;
    auto pos = target_device.find('.');
    if (pos != std::string::npos) {
        cleartarget_device = target_device.substr(0, pos);
        deviceID =  target_device.substr(pos + 1,  target_device.size());
    }
    // sw plugins are not requested to support `ov::available_devices` property
    if (!sw_plugin_in_target_device(target_device)) {
        auto deviceIDs = ie.get_property(cleartarget_device, ov::available_devices);
        if (std::find(deviceIDs.begin(), deviceIDs.end(), deviceID) == deviceIDs.end()) {
            GTEST_FAIL();
        }
    }

    OV_ASSERT_NO_THROW(ie.set_property(target_device, ov::enable_profiling(true)));
    bool value = false;
    OV_ASSERT_NO_THROW(value = ie.get_property(target_device, ov::enable_profiling));
    ASSERT_TRUE(value);
}


TEST_P(OVClassSetDevicePriorityConfigPropsTest, SetConfigAndCheckGetConfigNoThrow) {
    ov::Core ie = ov::test::utils::create_core();
    std::string devicePriority;
    OV_ASSERT_NO_THROW(ie.set_property(target_device, configuration));
    OV_ASSERT_NO_THROW(devicePriority = ie.get_property(target_device, ov::device::priorities));
    ASSERT_EQ(devicePriority, configuration[ov::device::priorities.name()].as<std::string>());
}

TEST_P(OVGetMetricPropsTest, GetMetricAndPrintNoThrow_AVAILABLE_DEVICES) {
    ov::Core ie = ov::test::utils::create_core();
    std::vector<std::string> device_ids;

    OV_ASSERT_NO_THROW(device_ids = ie.get_property(target_device, ov::available_devices));
    ASSERT_GT(device_ids.size(), 0);

    for (auto&& device_id : device_ids) {
        std::string full_name;
        OV_ASSERT_NO_THROW(full_name = ie.get_property(target_device, ov::device::full_name, ov::device::id(device_id)));
        ASSERT_FALSE(full_name.empty());
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVGetMetricPropsTest, GetMetriDeviceFullNameWithoutAdditionalTerminatorChars) {
    ov::Core core = ov::test::utils::create_core();
    auto supported_properties = core.get_property(target_device, ov::supported_properties);
    if (util::contains(supported_properties, ov::device::full_name)) {
        std::string full_name;
        OV_ASSERT_NO_THROW(full_name = core.get_property(target_device, ov::device::full_name));
        EXPECT_EQ(full_name.size(), strlen(full_name.c_str()));
    }
}

TEST_P(OVClassCompileModelAndCheckSecondaryPropertiesTest, CompileModelAndCheckSecondaryPropertiesTest) {
    ov::Core ie = ov::test::utils::create_core();
    ov::CompiledModel model;
    OV_ASSERT_NO_THROW(model = ie.compile_model(actualNetwork, target_device, configuration));
    ov::AnyMap property = configuration;
    ov::AnyMap::iterator it = configuration.end();
    // device properties in form ov::device::properties(DEVICE, ...) has the first priority
    for (it = configuration.begin(); it != configuration.end(); it++) {
        if ((it->first.find(ov::device::properties.name()) != std::string::npos) &&
            (it->first != ov::device::properties.name())) {
            break;
        }
    }
    if (it != configuration.end()) {
        // DEVICE_PROPERTIES_<DEVICE_NAME> found
        property = it->second.as<ov::AnyMap>();
    } else {
        // search for DEVICE_PROPERTIES
        it = configuration.find(ov::device::properties.name());
        ASSERT_TRUE(it != configuration.end());
        property = it->second.as<ov::AnyMap>().begin()->second.as<ov::AnyMap>();
        if (it == configuration.end()) {
            it = configuration.find(ov::num_streams.name());
        }
    }
    ASSERT_TRUE(property.count(ov::num_streams.name()));
    auto actual = property.at(ov::num_streams.name()).as<int32_t>();
    ov::Any value;
    //AutoExcutableNetwork GetMetric() does not support key ov::num_streams
    OV_ASSERT_NO_THROW(value = model.get_property(ov::num_streams.name()));
    int32_t expect = value.as<int32_t>();
    ASSERT_EQ(actual, expect);
}

TEST_P(OVGetMetricPropsTest, GetMetricAndPrintNoThrow_OPTIMIZATION_CAPABILITIES) {
    ov::Core ie = ov::test::utils::create_core();
    std::vector<std::string> capabilities;
    OV_ASSERT_NO_THROW(capabilities = ie.get_property(target_device, ov::device::capabilities));
    std::cout << "Optimization capabilities: " << std::endl;
    std::vector<std::string> possible_capabilities{ov::device::capability::FP32, ov::device::capability::BF16,
                                                   ov::device::capability::FP16, ov::device::capability::INT8,
                                                   ov::device::capability::INT16, ov::device::capability::BIN,
                                                   ov::device::capability::WINOGRAD, ov::device::capability::EXPORT_IMPORT};
    for (auto&& capability : capabilities) {
        ASSERT_TRUE(std::find(possible_capabilities.begin(), possible_capabilities.end(), capability) != possible_capabilities.end());
        std::cout << capability << std::endl;
    }
    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVGetMetricPropsOptionalTest, GetMetricAndPrintNoThrow_RANGE_FOR_ASYNC_INFER_REQUESTS) {
    ov::Core ie = ov::test::utils::create_core();
    unsigned int start{0}, end{0}, step{0};

    OV_ASSERT_NO_THROW(std::tie(start, end, step) = ie.get_property(target_device, ov::range_for_async_infer_requests));

    std::cout << "Range for async infer requests: " << std::endl
    << start << std::endl
    << end << std::endl
    << step << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    ASSERT_GE(step, 1u);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVGetMetricPropsOptionalTest, GetMetricAndPrintNoThrow_RANGE_FOR_STREAMS) {
    ov::Core ie = ov::test::utils::create_core();
    unsigned int start = 0, end = 0;

    OV_ASSERT_NO_THROW(std::tie(start, end) = ie.get_property(target_device, ov::range_for_streams));

    std::cout << "Range for streams: " << std::endl
    << start << std::endl
    << end << std::endl
    << std::endl;

    ASSERT_LE(start, end);
    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassSeveralDevicesTestDefaultCore, DefaultCoreSeveralDevicesNoThrow) {
    ov::Core ie;

    std::string clear_target_device;
    auto pos = target_devices.begin()->find('.');
    if (pos != std::string::npos) {
        clear_target_device = target_devices.begin()->substr(0, pos);
    }
    std::vector<std::string> deviceIDs;
    if (sw_plugin_in_target_device(clear_target_device)) {
        deviceIDs = {clear_target_device};
    } else {
        deviceIDs = ie.get_property(clear_target_device, ov::available_devices);
    }
    if (deviceIDs.size() < target_devices.size())
        GTEST_FAIL() << "Incorrect Device ID" << std::endl;

    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(ie.set_property(target_devices[i], ov::enable_profiling(true)));
    }
    bool res;
    for (size_t i = 0; i < target_devices.size(); ++i) {
        OV_ASSERT_NO_THROW(res = ie.get_property(target_devices[i], ov::enable_profiling));
        ASSERT_TRUE(res);
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
