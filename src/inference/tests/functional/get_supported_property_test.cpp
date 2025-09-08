// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

using TestParam = std::tuple<ov::AnyMap, ov::AnyMap, bool>;
class GetPropertyTest : public ::testing::TestWithParam<TestParam> {
public:
    ov::Core core;

    void SetUp() override {
        m_properties = std::get<0>(GetParam());
        m_expected_properties = std::get<1>(GetParam());
        m_keep_core_property = std::get<2>(GetParam());
    }

    void reg_plugin(ov::Core& core, std::shared_ptr<ov::IPlugin>& plugin) {
        std::string libraryPath = ov::test::utils::get_mock_engine_path();
        if (!m_so)
            m_so = ov::util::load_shared_object(libraryPath.c_str());
        std::function<void(ov::IPlugin*)> injectProxyEngine =
            ov::test::utils::make_std_function<void(ov::IPlugin*)>(m_so, "InjectPlugin");

        injectProxyEngine(plugin.get());
        core.register_plugin(ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                                                std::string("mock_engine") + OV_BUILD_POSTFIX),
                             m_plugin_name);
        m_mock_plugin = plugin;
    }

    void TearDown() override {
        core.unload_plugin(m_plugin_name);
    }

protected:
    std::shared_ptr<void> m_so;
    std::shared_ptr<ov::IPlugin> m_mock_plugin;
    ov::AnyMap m_properties;
    ov::AnyMap m_expected_properties;
    bool m_keep_core_property;
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
    auto plugin = std::make_shared<ov::test::utils::MockPlugin>();
    std::shared_ptr<ov::IPlugin> base_plugin = plugin;
    reg_plugin(core, base_plugin);
    core.get_property(m_plugin_name, ov::supported_properties);
    std::map<std::string, std::string> config;
    auto actual_output =
        m_mock_plugin->get_core()->get_supported_property(m_plugin_name, m_properties, m_keep_core_property);
    for (auto& iter : m_expected_properties) {
        ASSERT_TRUE(actual_output.find(iter.first) != actual_output.end());
        ASSERT_EQ(actual_output.find(iter.first)->second, iter.second);
    }
    for (auto& iter : m_properties) {
        if (m_expected_properties.find(iter.first) == m_expected_properties.end()) {
            ASSERT_TRUE(actual_output.find(iter.first) == actual_output.end());
        }
    }
}

static const std::vector<TestParam> test_variants = {
    TestParam{ov::AnyMap({ov::hint::allow_auto_batching(false), ov::num_streams(2)}),
              ov::AnyMap({ov::hint::allow_auto_batching(false), ov::num_streams(2)}),
              true},
    TestParam{ov::AnyMap({ov::auto_batch_timeout(0), ov::enable_profiling(false)}),
              ov::AnyMap({ov::auto_batch_timeout(0)}),
              true},
    TestParam{ov::AnyMap({ov::cache_dir("test"), ov::force_tbb_terminate(false)}),
              ov::AnyMap({ov::cache_dir("test"), ov::force_tbb_terminate(false)}),
              true},
    TestParam{ov::AnyMap({ov::cache_dir("test"),
                          ov::device::properties("MOCK_HARDWARE", ov::num_streams(2), ov::enable_profiling(true))}),
              ov::AnyMap({ov::cache_dir("test"), ov::num_streams(2)}),
              true},
    TestParam{ov::AnyMap({ov::num_streams(2)}), ov::AnyMap({ov::num_streams(2)}), false},
    TestParam{ov::AnyMap({ov::cache_dir("test")}), ov::AnyMap({}), false},
    TestParam{ov::AnyMap({ov::hint::allow_auto_batching(false)}), ov::AnyMap({}), false},
    TestParam{ov::AnyMap({ov::auto_batch_timeout(0)}), ov::AnyMap({}), false},
    TestParam{ov::AnyMap({ov::force_tbb_terminate(false)}), ov::AnyMap({}), false},
};

INSTANTIATE_TEST_SUITE_P(GetSupportedPropertyTest,
                         GetPropertyTest,
                         ::testing::ValuesIn(test_variants),
                         getTestCaseName);

TEST(PropertyTest, SetCacheDirPropertyCoreNoThrow) {
    ov::Core core;

    // Cache_dir property test
    ov::Any value;
    OV_ASSERT_NO_THROW(core.set_property(ov::cache_dir("./tmp_cache_dir")));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::cache_dir.name()));
    EXPECT_EQ(value.as<std::string>(), std::string("./tmp_cache_dir"));
}

TEST(PropertyTest, SetTBBForceTerminatePropertyCoreNoThrow) {
    ov::Core core;

    bool value = true;
    OV_ASSERT_NO_THROW(core.set_property(ov::force_tbb_terminate(false)));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::force_tbb_terminate.name()).as<bool>());
    EXPECT_FALSE(value);
    OV_ASSERT_NO_THROW(core.set_property(ov::force_tbb_terminate(true)));
    OV_ASSERT_NO_THROW(value = core.get_property(ov::force_tbb_terminate.name()).as<bool>());
    EXPECT_TRUE(value);
}

TEST(PropertyTest, GetUnsupportedPropertyCoreThrow) {
    ov::Core core;

    // Unsupported property test
    ASSERT_THROW(core.get_property("unsupported_property"), ov::Exception);
}
