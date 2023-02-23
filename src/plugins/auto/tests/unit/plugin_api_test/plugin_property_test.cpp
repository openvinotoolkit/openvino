// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "test_utils/mock_auto_device_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "test_utils/param_helper.hpp"
#include "common_test_utils/file_utils.hpp"
#include "openvino/runtime/core.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;
using namespace MockMultiDevicePlugin;
using namespace MockMultiDevice;

using pluginConfigParams = std::tuple<
                        std::string,                           // test MULTI or AUTO
                        std::map<ov::PropertyName, ov::Any>    // property to test
                        >;
class SetGetConfigAndGetMetricTest : public ::testing::TestWithParam<pluginConfigParams> {
public:
    std::shared_ptr<NiceMock<MockICore>> core;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;
    std::string pluginToTest;
    std::map<ov::PropertyName, ov::Any> properties;

public:
    static std::string getTestCaseName(testing::TestParamInfo<pluginConfigParams> obj) {
        std::string pluginname;
        std::map<ov::PropertyName, ov::Any> configuration;
        std::tie(pluginname, configuration) = obj.param;
        std::ostringstream result;
        result << pluginname;
        for (const std::pair<ov::PropertyName, ov::Any>& iter : configuration) {
            if (!iter.second.empty())
                result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
            else
                result << "_" << iter.first << "_";
        }
        return result.str();
    }

    void TearDown() override {
        core.reset();
        plugin.reset();
    }

    void SetUp() override {
        std::tie(pluginToTest, properties) = GetParam();
        core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        plugin = std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>>(new NiceMock<MockMultiDeviceInferencePlugin>());
        plugin->SetCore(core);
        plugin->SetName(pluginToTest);
    }
};

using SetGetValidConfigMetricTest = SetGetConfigAndGetMetricTest;
using SetGetInValidConfigMetricTest = SetGetConfigAndGetMetricTest;
using GetDefaultConfigMetricTest = SetGetConfigAndGetMetricTest;

TEST_P(SetGetValidConfigMetricTest, SetGetCorrectKeyAndValueTestCase) {
    std::vector<ov::PropertyName> supported_properties;
    auto temp = plugin->GetMetric(ov::supported_properties.name(), {});
    ASSERT_NO_THROW(supported_properties = temp.as<std::vector<ov::PropertyName>>());
    for (const auto& property_item : properties) {
        auto supported = ov::util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_value;
        if (property_item.first.is_mutable()) {
            ASSERT_NO_THROW(default_value = plugin->GetConfig(property_item.first, {}));
        } else {
            ASSERT_NO_THROW(default_value = plugin->GetMetric(property_item.first, {}));
            if (!property_item.second.empty()) {
                ASSERT_THROW(plugin->SetConfig({{property_item.first, property_item.second.as<std::string>()}}), IE::Exception);
            }
        }
        if (property_item.first.is_mutable() && !property_item.second.empty()) {
            ASSERT_NO_THROW(plugin->SetConfig({{property_item.first, property_item.second.as<std::string>()}}));
            ov::Any new_property_value;
            ASSERT_NO_THROW(new_property_value = plugin->GetConfig(property_item.first, {}));
            EXPECT_EQ(new_property_value.as<std::string>(), property_item.second.as<std::string>());
        }
    }
}

TEST_P(GetDefaultConfigMetricTest, CanGetDefaultValue) {
    std::vector<ov::PropertyName> supported_properties;
    auto temp = plugin->GetMetric(ov::supported_properties.name(), {});
    ASSERT_NO_THROW(supported_properties = temp.as<std::vector<ov::PropertyName>>());
    for (const auto& property_item : properties) {
        auto supported = ov::util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;
        ov::Any default_value;
        if (property_item.first.is_mutable()) {
            ASSERT_NO_THROW(default_value = plugin->GetConfig(property_item.first, {}));
            ASSERT_FALSE(default_value.empty());
            ASSERT_EQ(default_value.as<std::string>(), property_item.second.as<std::string>());
        } else {
            ASSERT_NO_THROW(default_value = plugin->GetMetric(property_item.first, {}));
        }
    }
}

TEST_P(SetGetInValidConfigMetricTest, SetGetIncorrectKeyOrValueTestCase) {
    std::vector<ov::PropertyName> supported_properties;
    for (const auto& property_item : properties) {
        ov::Any default_value;
        if (!property_item.first.is_mutable())
            ASSERT_THROW(default_value = plugin->GetMetric(property_item.first, {}), IE::Exception);
        else if (property_item.second.empty())
            ASSERT_THROW(default_value = plugin->GetConfig(property_item.first, {}), IE::Exception);
        else
            ASSERT_THROW(plugin->SetConfig({{property_item.first, property_item.second.as<std::string>()}}), IE::Exception);
    }
}

std::vector<std::string> test_plugin_targets = {"MULTI", "AUTO"};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SetGetValidConfigMetricTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_plugin_targets),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::VALID>>())->get_params())),
                         SetGetConfigAndGetMetricTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         SetGetInValidConfigMetricTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_plugin_targets),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::INVALID>>())->get_params())),
                         SetGetConfigAndGetMetricTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         GetDefaultConfigMetricTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_plugin_targets),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::DEFAULT>>())->get_params())),
                         GetDefaultConfigMetricTest::getTestCaseName);
/*
class CoreIntegrationPropertyTest : public ::testing::TestWithParam<pluginConfigParams> {
public:
    std::shared_ptr<void> sharedObjectLoader;
    std::function<void(IInferencePlugin*)> injectProxyEngine;
    std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>> plugin;
    std::string pluginToTest;
    std::map<ov::PropertyName, ov::Any> properties;

private:
    template <class T>
    std::function<T> make_std_function(const std::string& functionName) {
        std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(sharedObjectLoader, functionName.c_str())));
        return ptr;
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<pluginConfigParams> obj) {
        std::string pluginname;
        std::map<ov::PropertyName, ov::Any> configuration;
        std::tie(pluginname, configuration) = obj.param;
        std::ostringstream result;
        result << pluginname;
        for (const std::pair<ov::PropertyName, ov::Any>& iter : configuration) {
            if (!iter.second.empty())
                result << "_" << iter.first << "_" << iter.second.as<std::string>() << "_";
            else
                result << "_" << iter.first << "_";
        }
        return result.str();
    }
    static std::string get_mock_engine_path() {
        std::string mockEngineName("mock_engine");
        return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                  mockEngineName + IE_BUILD_POSTFIX);
    }

    void TearDown() override {
        plugin.reset();
    }

    void SetUp() override {
        plugin = std::shared_ptr<NiceMock<MockMultiDeviceInferencePlugin>>(new NiceMock<MockMultiDeviceInferencePlugin>());
        std::string libraryPath = get_mock_engine_path();
        sharedObjectLoader = ov::util::load_shared_object(libraryPath.c_str());
        injectProxyEngine = make_std_function<void(InferenceEngine::IInferencePlugin*)>("InjectProxyEngine");

        std::tie(pluginToTest, properties) = GetParam();
        plugin->SetName(pluginToTest);
    }
};

TEST_P(CoreIntegrationPropertyTest, SetGetCorrectKeyAndValueTestCase) {
    ov::Core core;
    injectProxyEngine(plugin.get());
    //core.unload_plugin(pluginToTest);
    core.register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                             std::string("mock_engine") + IE_BUILD_POSTFIX),
                          pluginToTest);
    std::vector<ov::PropertyName> supported_properties = core.get_property(pluginToTest, ov::supported_properties);
    for (const auto& property_item : properties) {
        auto supported = ov::util::contains(supported_properties, property_item.first);
        ASSERT_TRUE(supported) << "property is not supported: " << property_item.first;

        ov::Any default_value;
        if (property_item.first.is_mutable()) {
            ASSERT_NO_THROW(default_value = core.get_property(pluginToTest, property_item.first));
        } else {
            ASSERT_NO_THROW(default_value = core.get_property(pluginToTest, property_item.first));
            //if (!property_item.second.empty())
                //ASSERT_THROW(core.set_property(pluginToTest, ov::AnyMap{property_item.first.get_name(), property_item.second}), IE::Exception);
        }
        /*if (property_item.first.is_mutable() && !property_item.second.empty()) {
            ASSERT_NO_THROW(plugin->SetConfig({{property_item.first, property_item.second.as<std::string>()}}));
            ov::Any new_property_value;
            ASSERT_NO_THROW(new_property_value = plugin->GetConfig(property_item.first, {}));
            EXPECT_EQ(new_property_value.as<std::string>(), property_item.second.as<std::string>());
        }
    }
    core.unload_plugin(pluginToTest);
}
std::vector<std::string> test_targets_core = {"MOCK_MULTI", "MOCK_AUTO"};
INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         CoreIntegrationPropertyTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(test_targets_core),
                            ::testing::ValuesIn((std::make_shared<ParamSet<TestParamType::VALID>>())->get_params())),
                         CoreIntegrationPropertyTest::getTestCaseName);
                         */