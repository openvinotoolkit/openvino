// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_core.hpp"
#include "ie_icore.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace {
std::string get_mock_engine_path() {
    std::string mockEngineName("mock_engine");
    return ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                              mockEngineName + IE_BUILD_POSTFIX);
}
ov::PropertyName RO_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
}
ov::PropertyName RW_property(const std::string& propertyName) {
    return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
}
template <class T>
std::function<T> make_std_function(const std::shared_ptr<void> so, const std::string& functionName) {
    std::function<T> ptr(reinterpret_cast<T*>(ov::util::get_symbol(so, functionName.c_str())));
    return ptr;
}

}  // namespace

class MockHardWarePlugin : public InferenceEngine::IInferencePlugin {
public:
    void SetConfig(const std::map<std::string, std::string>& config) override {
        for (const auto& it : config) {
            if (it.first == ov::num_streams.name())
                num_streams = std::stoi(it.second);
            else
                OPENVINO_THROW(GetName(), " set config: " + it.first);
        }
    }
    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        const std::vector<ov::PropertyName> roProperties{
            RO_property(ov::supported_properties.name()),
        };
        // the whole config is RW before network is loaded.
        const std::vector<ov::PropertyName> rwProperties{
            RW_property(ov::num_streams.name()),
        };
        if (name == ov::supported_properties) {
            std::vector<ov::PropertyName> supportedProperties;
            supportedProperties.reserve(roProperties.size() + rwProperties.size());
            supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
            supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());

            return decltype(ov::supported_properties)::value_type(supportedProperties);
        } else if (name == "SUPPORTED_METRICS") {  //
            std::vector<std::string> configs;
            for (const auto& property : roProperties) {
                configs.emplace_back(property);
            }
            return configs;
        } else if (name == "SUPPORTED_CONFIG_KEYS") {
            std::vector<std::string> configs;
            for (const auto& property : rwProperties) {
                configs.emplace_back(property);
            }
            return configs;
        } else if (name == ov::num_streams.name()) {
            return num_streams;
        }
        return "";
    }

private:
    int32_t num_streams{0};
};

using TestParam = std::tuple<ov::AnyMap, ov::AnyMap>;
class GetPropertyTest : public ::testing::TestWithParam<TestParam> {
public:
    ov::Core core;

    void SetUp() override {
        m_properties = std::get<0>(GetParam());
        m_expected_properties = std::get<1>(GetParam());
    }

    void reg_plugin(ov::Core& core, std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) {
        std::string libraryPath = get_mock_engine_path();
        if (!m_so)
            m_so = ov::util::load_shared_object(libraryPath.c_str());
        std::function<void(InferenceEngine::IInferencePlugin*)> injectProxyEngine =
            make_std_function<void(InferenceEngine::IInferencePlugin*)>(m_so, "InjectProxyEngine");

        injectProxyEngine(plugin.get());
        core.register_plugin(ov::util::make_plugin_library_name(CommonTestUtils::getExecutableDirectory(),
                                                                std::string("mock_engine") + IE_BUILD_POSTFIX),
                             m_plugin_name);
        m_mock_plugin = plugin;
    }

    void TearDown() override {
        core.unload_plugin(m_plugin_name);
    }

protected:
    std::shared_ptr<void> m_so;
    std::shared_ptr<InferenceEngine::IInferencePlugin> m_mock_plugin;
    ov::AnyMap m_properties;
    ov::AnyMap m_expected_properties;
    std::string m_plugin_name{"MOCK_HARDWARE"};
};

static std::string getTestCaseName(const testing::TestParamInfo<TestParam>& obj) {
    ov::AnyMap properties = std::get<0>(obj.param);
    ov::AnyMap expected_properties = std::get<1>(obj.param);
    std::ostringstream result;
    result << "query_property_";
    for (auto& item : properties) {
        result << item.first << "_" << item.second.as<std::string>() << "_";
    }
    result << "expected_property_";
    for (auto& item : expected_properties) {
        result << item.first << "_" << item.second.as<std::string>() << "_";
    }
    auto name = result.str();
    name.pop_back();
    return name;
}

TEST_P(GetPropertyTest, canGenerateCorrectPropertyList) {
    auto plugin = std::make_shared<MockHardWarePlugin>();
    std::shared_ptr<InferenceEngine::IInferencePlugin> base_plugin = plugin;
    reg_plugin(core, base_plugin);
    core.get_property(m_plugin_name, ov::supported_properties);
    std::map<std::string, std::string> config;
    for (auto&& value : m_properties) {
        config.emplace(value.first, value.second.as<std::string>());
    }
    auto actual_output = m_mock_plugin->GetCore()->GetSupportedConfig(m_plugin_name, config);
    for (auto& iter : m_expected_properties) {
        ASSERT_TRUE(actual_output.find(iter.first) != actual_output.end());
        ASSERT_EQ(actual_output.find(iter.first)->second, iter.second.as<std::string>());
    }
    for (auto& iter : m_properties) {
        if (m_expected_properties.find(iter.first) == m_expected_properties.end()) {
            ASSERT_TRUE(actual_output.find(iter.first) == actual_output.end());
        }
    }
}

static const std::vector<TestParam> test_variants = {
    TestParam{ov::AnyMap({ov::hint::allow_auto_batching(false), ov::num_streams(2)}),
              ov::AnyMap({ov::hint::allow_auto_batching(false), ov::num_streams(2)})},
    TestParam{ov::AnyMap({ov::auto_batch_timeout(0), ov::enable_profiling(false)}),
              ov::AnyMap({ov::auto_batch_timeout(0)})},
    TestParam{ov::AnyMap({ov::cache_dir("test"), ov::force_tbb_terminate(false)}),
              ov::AnyMap({ov::cache_dir("test"), ov::force_tbb_terminate(false)})},
    TestParam{ov::AnyMap({ov::cache_dir("test"),
                          ov::device::properties("MOCK_HARDWARE", ov::num_streams(2), ov::enable_profiling(true))}),
              ov::AnyMap({ov::cache_dir("test"), ov::num_streams(2)})},
};

INSTANTIATE_TEST_SUITE_P(GetSupportedPropertyTest,
                         GetPropertyTest,
                         ::testing::ValuesIn(test_variants),
                         getTestCaseName);