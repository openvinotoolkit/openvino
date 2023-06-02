// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

namespace {
std::string get_string_value(const ov::Any& value) {
    if (value.empty()) {
        return "Empty";
    } else {
        return value.as<std::string>();
    }
}
}  // namespace

TEST_F(ProxyTests, get_property_on_default_uninit_device) {
    const std::string dev_name = "MOCK";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
}

TEST_F(ProxyTests, set_property_for_fallback_device) {
    const std::string dev_name = "MOCK.1";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::device::properties("BDE", ov::enable_profiling(true)));
    EXPECT_EQ(false, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_property_for_primary_device) {
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_property_for_primary_device_full_name) {
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC.abc_b", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, get_property_on_default_device) {
    const std::string dev_name = "MOCK";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(10, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("000102030405060708090a0b0c0d0e0f", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            EXPECT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(4, mutable_pr);
}

TEST_F(ProxyTests, get_property_on_mixed_device) {
    const std::string dev_name = "MOCK.1";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(10, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("00020406080a0c0e10121416181a1c1e", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            EXPECT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            core.get_property(dev_name, property);
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(4, mutable_pr);
}

TEST_F(ProxyTests, get_property_on_specified_device) {
    const std::string dev_name = "MOCK.3";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(9, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            EXPECT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}

TEST_F(ProxyTests, get_property_for_changed_default_device) {
    const std::string dev_name = "MOCK";
    core.set_property(dev_name, ov::device::id(3));
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(9, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            EXPECT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}
