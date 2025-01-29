// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/plugin_config.hpp"

#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>

using namespace ::testing;
using namespace ov;

static constexpr Property<float, PropertyMutability::RW> unsupported_property{"UNSUPPORTED_PROPERTY"};
static constexpr Property<bool, PropertyMutability::RW> bool_property{"BOOL_PROPERTY"};
static constexpr Property<int32_t, PropertyMutability::RW> int_property{"INT_PROPERTY"};
static constexpr Property<std::string, PropertyMutability::RW> high_level_property{"HIGH_LEVEL_PROPERTY"};
static constexpr Property<std::string, PropertyMutability::RW> low_level_property{"LOW_LEVEL_PROPERTY"};
static constexpr Property<int64_t, PropertyMutability::RW> release_internal_property{"RELEASE_INTERNAL_PROPERTY"};

#ifdef ENABLE_DEBUG_CAPS
static constexpr Property<int64_t, PropertyMutability::RW> debug_property{"DEBUG_PROPERTY"};
static constexpr Property<int32_t, PropertyMutability::RW> debug_global_property{"DEBUG_GLOBAL_PROPERTY"};
#endif

namespace {
const std::string test_config_path = "test_debug_config_path.json";
const std::string device_name = "SOME_DEVICE";

void dump_config(const std::string& filename, const std::string& config_content) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Can't save config file \"" + filename + "\".");
    }

    ofs << config_content;
}

}  // namespace

struct EmptyTestConfig : public ov::PluginConfig {
    std::vector<std::string> get_supported_properties() const {
        std::vector<std::string> supported_properties;
        for (const auto& kv : m_options_map) {
            supported_properties.push_back(kv.first);
        }
        return supported_properties;
    }
};

struct NotEmptyTestConfig : public ov::PluginConfig {
    NotEmptyTestConfig() {
    #define OV_CONFIG_LOCAL_OPTION(...) EXPAND(OV_CONFIG_OPTION_MAPPING(__VA_ARGS__))
    #define OV_CONFIG_GLOBAL_OPTION(...) EXPAND(OV_CONFIG_OPTION_MAPPING(__VA_ARGS__))
        OV_CONFIG_RELEASE_OPTION(, bool_property, true, "")
        OV_CONFIG_RELEASE_OPTION(, int_property, -1, "")
        OV_CONFIG_RELEASE_OPTION(, high_level_property, "", "")
        OV_CONFIG_RELEASE_OPTION(, low_level_property, "", "")
        OV_CONFIG_RELEASE_INTERNAL_OPTION(, release_internal_property, 1, "")
        OV_CONFIG_DEBUG_OPTION(, debug_property, 2, "")
        OV_CONFIG_DEBUG_GLOBAL_OPTION(, debug_global_property, 4, "")
    #undef OV_CONFIG_LOCAL_OPTION
    #undef OV_CONFIG_GLOBAL_OPTION
    }

    NotEmptyTestConfig(const NotEmptyTestConfig& other) : NotEmptyTestConfig() {
        m_user_properties = other.m_user_properties;
        for (const auto& kv : other.m_options_map) {
            m_options_map.at(kv.first)->set_any(kv.second->get_any());
        }
    }

    #define OV_CONFIG_LOCAL_OPTION(...) EXPAND(OV_CONFIG_DECLARE_LOCAL_OPTION(__VA_ARGS__))  EXPAND(OV_CONFIG_DECLARE_LOCAL_GETTER(__VA_ARGS__))
    #define OV_CONFIG_GLOBAL_OPTION(...) EXPAND(OV_CONFIG_DECLARE_GLOBAL_OPTION(__VA_ARGS__))  EXPAND(OV_CONFIG_DECLARE_GLOBAL_GETTER(__VA_ARGS__))
        OV_CONFIG_RELEASE_OPTION(, bool_property, true, "")
        OV_CONFIG_RELEASE_OPTION(, int_property, -1, "")
        OV_CONFIG_RELEASE_OPTION(, high_level_property, "", "")
        OV_CONFIG_RELEASE_OPTION(, low_level_property, "", "")
        OV_CONFIG_RELEASE_INTERNAL_OPTION(, release_internal_property, 1, "")
        OV_CONFIG_DEBUG_OPTION(, debug_property, 2, "")
        OV_CONFIG_DEBUG_GLOBAL_OPTION(, debug_global_property, 4, "")
    #undef OV_CONFIG_LOCAL_OPTION
    #undef OV_CONFIG_GLOBAL_OPTION

    std::vector<std::string> get_supported_properties() const {
        std::vector<std::string> supported_properties;
        for (const auto& kv : m_options_map) {
            supported_properties.push_back(kv.first);
        }
        return supported_properties;
    }

    void finalize_impl(const IRemoteContext* context) override {
        if (!is_set_by_user(low_level_property)) {
            m_low_level_property.value = m_high_level_property.value;
        }
#ifdef ENABLE_DEBUG_CAPS
        apply_config_options(device_name, test_config_path);
#endif
    }

    void apply_model_specific_options(const IRemoteContext* context, const ov::Model& model) override {
        apply_rt_info_property(high_level_property, model.get_rt_info<ov::AnyMap>("runtime_options"));
    }

    using ov::PluginConfig::get_option_ptr;
    using ov::PluginConfig::is_set_by_user;
};

#define OV_CONFIG_LOCAL_OPTION(...)
#define OV_CONFIG_GLOBAL_OPTION(PropertyNamespace, PropertyVar, Visibility, ...) \
    ConfigOption<decltype(PropertyNamespace::PropertyVar)::value_type, Visibility> NotEmptyTestConfig::m_ ## PropertyVar{GET_EXCEPT_LAST(__VA_ARGS__)};

    OV_CONFIG_DEBUG_GLOBAL_OPTION(, debug_global_property, 4, "")

#undef OV_CONFIG_LOCAL_OPTION
#undef OV_CONFIG_GLOBAL_OPTION

TEST(plugin_config, can_create_empty_config) {
    ASSERT_NO_THROW(
        EmptyTestConfig cfg;
        ASSERT_EQ(cfg.get_supported_properties().size(), 0);
    );
}

TEST(plugin_config, can_create_not_empty_config) {
#ifdef ENABLE_DEBUG_CAPS
       size_t expected_options_num = 7;
#else
       size_t expected_options_num = 5;
#endif
    ASSERT_NO_THROW(
        NotEmptyTestConfig cfg;
        ASSERT_EQ(cfg.get_supported_properties().size(), expected_options_num);
    );
}

TEST(plugin_config, can_set_get_property) {
    NotEmptyTestConfig cfg;
    ASSERT_NO_THROW(cfg.get_bool_property());
    ASSERT_EQ(cfg.get_bool_property(), true);
    ASSERT_NO_THROW(cfg.set_property(bool_property(false)));
    ASSERT_EQ(cfg.get_bool_property(), false);
}

TEST(plugin_config, throw_for_unsupported_property) {
    NotEmptyTestConfig cfg;
    ASSERT_ANY_THROW(cfg.get_property(unsupported_property.name()));
    ASSERT_ANY_THROW(cfg.set_property(unsupported_property(10.0f)));
}

TEST(plugin_config, can_direct_access_to_properties) {
    NotEmptyTestConfig cfg;
    ASSERT_EQ(cfg.m_int_property.value, cfg.get_int_property());
    ASSERT_NO_THROW(cfg.set_user_property(int_property(1)));
    ASSERT_EQ(cfg.m_int_property.value, -1); // user property doesn't impact member value until finalize() is called

    cfg.m_int_property.value = 2;
    ASSERT_EQ(cfg.get_int_property(), 1); // stil 1 as user property was set previously
}

TEST(plugin_config, finalization_updates_member) {
    NotEmptyTestConfig cfg;
    ASSERT_NO_THROW(cfg.set_user_property(bool_property(false)));
    ASSERT_EQ(cfg.m_bool_property.value, true); // user property doesn't impact member value until finalize() is called

    cfg.finalize(nullptr, {});

    ASSERT_EQ(cfg.m_bool_property.value, false); // now the value has changed
}

TEST(plugin_config, get_property_before_finalization_returns_user_property_if_set) {
    NotEmptyTestConfig cfg;

    ASSERT_EQ(cfg.get_bool_property(), true);  // default value
    ASSERT_EQ(cfg.m_bool_property.value, true);  // default value

    cfg.m_bool_property.value = false; // update member directly
    ASSERT_EQ(cfg.get_bool_property(), false);  // OK, return the class member value as no user property was set

    ASSERT_NO_THROW(cfg.set_user_property(bool_property(true)));
    ASSERT_TRUE(cfg.is_set_by_user(bool_property));
    ASSERT_EQ(cfg.get_bool_property(), true);  // now user property value is returned
    ASSERT_EQ(cfg.m_bool_property.value, false);  // but class member is not updated

    cfg.finalize(nullptr, {});
    ASSERT_EQ(cfg.get_bool_property(), cfg.m_bool_property.value);  // equal after finalization
    ASSERT_FALSE(cfg.is_set_by_user(bool_property)); // and user property is cleared
}

TEST(plugin_config, finalization_updates_dependant_properties) {
    NotEmptyTestConfig cfg;

    cfg.set_user_property(high_level_property("value1"));
    ASSERT_TRUE(cfg.is_set_by_user(high_level_property));
    ASSERT_FALSE(cfg.is_set_by_user(low_level_property));

    cfg.finalize(nullptr, {});
    ASSERT_EQ(cfg.m_high_level_property.value, "value1");
    ASSERT_EQ(cfg.m_low_level_property.value, "value1");
    ASSERT_FALSE(cfg.is_set_by_user(high_level_property));
    ASSERT_FALSE(cfg.is_set_by_user(low_level_property));
}

TEST(plugin_config, can_set_property_from_rt_info) {
    NotEmptyTestConfig cfg;

    RTMap rt_info = {
        {high_level_property.name(), "value1"},
        {int_property.name(), 10} // int_property is not applied from rt info
    };

    auto p1 = std::make_shared<ov::op::v0::Parameter>();
    auto r1 = std::make_shared<ov::op::v0::Result>(p1);
    ov::Model m(ov::OutputVector{r1}, ov::ParameterVector{p1});
    m.set_rt_info(rt_info, {"runtime_options"});

    // default values
    ASSERT_EQ(cfg.m_high_level_property.value, "");
    ASSERT_EQ(cfg.m_low_level_property.value, "");
    ASSERT_EQ(cfg.m_int_property.value, -1);

    cfg.finalize(nullptr, &m);

    ASSERT_EQ(cfg.m_high_level_property.value, "value1");
    ASSERT_EQ(cfg.m_low_level_property.value, "value1"); // dependant is updated too
    ASSERT_EQ(cfg.m_int_property.value, -1); // still default
}

TEST(plugin_config, can_copy_config) {
    NotEmptyTestConfig cfg1;

    cfg1.m_high_level_property.value = "value1";
    cfg1.m_low_level_property.value = "value2";
    cfg1.m_int_property.value = 1;
    cfg1.set_property(bool_property(false));

    NotEmptyTestConfig cfg2 = cfg1;
    ASSERT_EQ(cfg2.m_high_level_property.value, "value1");
    ASSERT_EQ(cfg2.m_low_level_property.value, "value2");
    ASSERT_EQ(cfg2.m_int_property.value, 1);
    ASSERT_EQ(cfg2.get_bool_property(), false); // ensure user properties are copied too

    // check that cfg1 modification doesn't impact a copy
    cfg1.set_property(high_level_property("value3"));
    cfg1.m_int_property.value = 3;
    ASSERT_EQ(cfg2.m_high_level_property.value, "value1");
    ASSERT_EQ(cfg2.m_int_property.value, 1);
}

TEST(plugin_config, set_property_throw_for_non_release_options) {
    NotEmptyTestConfig cfg;
    ASSERT_ANY_THROW(cfg.set_user_property({release_internal_property(10)}, OptionVisibility::RELEASE));
#ifdef ENABLE_DEBUG_CAPS
    ASSERT_ANY_THROW(cfg.set_user_property({debug_property(10)}, OptionVisibility::RELEASE));
#endif
}

TEST(plugin_config, visibility_is_correct) {
    NotEmptyTestConfig cfg;
    ASSERT_EQ(cfg.get_option_ptr(release_internal_property.name())->get_visibility(), OptionVisibility::RELEASE_INTERNAL);
    ASSERT_EQ(cfg.get_option_ptr(int_property.name())->get_visibility(), OptionVisibility::RELEASE);

#ifdef ENABLE_DEBUG_CAPS
    ASSERT_EQ(cfg.get_option_ptr(debug_property.name())->get_visibility(), OptionVisibility::DEBUG);
#endif
}

TEST(plugin_config, can_read_from_env_with_debug_caps) {
    try {
        NotEmptyTestConfig cfg;
        ASSERT_EQ(cfg.get_int_property(), -1);
        std::string env_var1 = "OV_INT_PROPERTY=10";
        ::putenv(env_var1.data());
        ASSERT_EQ(cfg.get_int_property(), -1); // env is applied after finalization only for build with debug caps

#ifdef ENABLE_DEBUG_CAPS
        std::string env_var2 = "OV_DEBUG_PROPERTY=20";
        ::putenv(env_var2.data());
        ASSERT_EQ(cfg.get_debug_property(), 2); // same for debug option
#endif

        cfg.finalize(nullptr, nullptr);

#ifdef ENABLE_DEBUG_CAPS
        ASSERT_EQ(cfg.get_int_property(), 10);
        ASSERT_EQ(cfg.get_debug_property(), 20);
#else
        ASSERT_EQ(cfg.get_int_property(), -1); // no effect
#endif
    } catch (std::exception&) {}

    ::unsetenv("OV_INT_PROPERTY");
#ifdef ENABLE_DEBUG_CAPS
    ::unsetenv("OV_DEBUG_PROPERTY");
#endif
}

TEST(plugin_config, can_read_from_config) {
    const std::filesystem::path filepath = test_config_path;
    try {
        NotEmptyTestConfig cfg;
        std::string config = "{\"SOME_DEVICE\":{\"DEBUG_PROPERTY\":\"20\",\"INT_PROPERTY\":\"10\"}}";

        dump_config(filepath.generic_string(), config);

        ASSERT_EQ(cfg.get_int_property(), -1); // config is applied after finalization only for build with debug caps
    #ifdef ENABLE_DEBUG_CAPS
        ASSERT_EQ(cfg.get_debug_property(), 2); // same for debug option
    #endif

        cfg.finalize(nullptr, nullptr);
    #ifdef ENABLE_DEBUG_CAPS
        ASSERT_EQ(cfg.get_int_property(), 10);
        ASSERT_EQ(cfg.get_debug_property(), 20);
    #else
        ASSERT_EQ(cfg.get_int_property(), -1); // no effect
    #endif
    } catch (std::exception&) { }

    std::filesystem::remove(filepath);
}

#ifdef ENABLE_DEBUG_CAPS
TEST(plugin_config, can_get_global_property) {
    NotEmptyTestConfig cfg;
    ASSERT_EQ(cfg.get_debug_global_property(), 4);
}

TEST(plugin_config, global_property_read_env_on_each_call) {
    try {
        ASSERT_EQ(NotEmptyTestConfig::get_debug_global_property(), 4);
        std::string env_var1 = "OV_DEBUG_GLOBAL_PROPERTY=10";
        ::putenv(env_var1.data());
        ASSERT_EQ(NotEmptyTestConfig::get_debug_global_property(), 10);

        std::string env_var2 = "OV_DEBUG_GLOBAL_PROPERTY=20";
        ::putenv(env_var2.data());
        ASSERT_EQ(NotEmptyTestConfig::get_debug_global_property(), 20);
    } catch (std::exception&) {}

    ::unsetenv("OV_DEBUG_GLOBAL_PROPERTY");
}
#endif
