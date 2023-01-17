// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/property_supervisor.hpp"

#include <gtest/gtest.h>

#include "ie_plugin_config.hpp"
#include "openvino/runtime/properties.hpp"

ov::Property<int, ov::PropertyMutability::RO> ro_property{"RO_PROPERTY"};
ov::Property<int, ov::PropertyMutability::RW> rw_property{"RW_PROPERTY"};
ov::Property<std::string, ov::PropertyMutability::RW> str_property{"RW_PROPERTY"};

TEST(OVPropertySupervisorTests, empty) {
    ov::PropertySupervisor properties;
    ASSERT_TRUE(properties.empty());
}

TEST(OVPropertySupervisorTests, bindROConstant) {
    ov::PropertySupervisor properties;
    properties.add(ro_property, 42);
    ASSERT_EQ(42, properties.get(ro_property));
    ASSERT_THROW(properties.set(ro_property.name(), 0), ov::Exception);
}

TEST(OVPropertySupervisorTests, bindRWCStr) {
    ov::PropertySupervisor properties;
    properties.add("rw_property", "42");
    ASSERT_EQ(std::string{"42"}, properties.get("rw_property").as<std::string>());
    ASSERT_NO_THROW(properties.set("rw_property", ""));
}

TEST(OVPropertySupervisorTests, bindRWStr) {
    ov::PropertySupervisor properties;
    properties.add(str_property, "42");
    ASSERT_EQ(std::string{"42"}, properties.get(str_property));
    ASSERT_THROW(properties.set("ro_property", ""), ov::Exception);
}

TEST(OVPropertySupervisorTests, bindROtoRWProperty) {
    ov::PropertySupervisor properties;
    properties.add(rw_property, 42, ov::PropertyMutability::RO);
    ASSERT_EQ(42, properties.get(rw_property));
    ASSERT_THROW(properties.set(rw_property(0)), ov::Exception);
}

TEST(OVPropertySupervisorTests, bindROGetter) {
    ov::PropertySupervisor properties;
    properties.add(ro_property, [] {
        return 42;
    });
    ASSERT_EQ(42, properties.get(ro_property));
    ASSERT_THROW(properties.set(ov::AnyMap{{ro_property.name(), 0}}), ov::Exception);
}

TEST(OVPropertySupervisorTests, bindRWGetterSetter) {
    ov::PropertySupervisor properties;
    int v = 0;
    properties.add(
        rw_property,
        [&] {
            return v;
        },
        [&](const int& x) {
            v = 42;
        });
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertySupervisorTests, bindRWRef) {
    ov::PropertySupervisor properties;
    int v = 0;
    properties.add(rw_property, std::ref(v));
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertySupervisorTests, bindRWStoredInternally) {
    ov::PropertySupervisor properties;
    properties.add(rw_property, 0);
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertySupervisorTests, precondition) {
    ov::PropertySupervisor properties;
    int num = 42;
    properties.add(rw_property.name(), num, [](const int& v) {
        OPENVINO_ASSERT(v < 42);
    });
    ASSERT_EQ(42, properties.get(rw_property));
    ASSERT_NO_THROW(properties.set(rw_property(41)));
    ASSERT_THROW(properties.set(rw_property(43)), ov::Exception);
    ASSERT_EQ(41, properties.get(rw_property));
}

TEST(OVPropertySupervisorTests, bindSubProperties) {
    ov::PropertySupervisor properties;
    ov::PropertySupervisor sub_properties;
    sub_properties.add(ro_property, 42);
    sub_properties.add(rw_property, 42);
    properties.add("sub", sub_properties);

    auto expected_supported_properties = std::vector<ov::PropertyName>{
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(SUPPORTED_METRICS),
        ov::supported_properties.name(),
        ro_property.name(),   // NOTE: AS this property name is unique
        rw_property.name()};  // it is represented with redused path thus it is situated in sub property set

    ASSERT_EQ(expected_supported_properties, properties.get(ov::supported_properties));

    ASSERT_EQ(42, properties.get(ov::device::properties("sub", ro_property)));
    properties.set(ov::device::properties("sub", rw_property(24)));
    ASSERT_EQ(24, properties.get(ov::device::properties("sub", rw_property)));
}

TEST(OVPropertySupervisorTests, ambiguousNameThrow) {
    ov::PropertySupervisor properties;
    properties.add(rw_property, 0);
    {
        ov::PropertySupervisor sub_properties;
        sub_properties.add(ro_property, 42);
        properties.add("sub0", sub_properties);
    }
    {
        ov::PropertySupervisor sub_properties;
        sub_properties.add(ro_property, 42);
        properties.add("sub1", sub_properties);
    }
    auto expected_supported_properties = std::vector<ov::PropertyName>{rw_property.name(),
                                                                       METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                                       METRIC_KEY(SUPPORTED_METRICS),
                                                                       ov::supported_properties.name(),
                                                                       std::string{"sub0."} + ro_property.name(),
                                                                       std::string{"sub1."} + ro_property.name()};

    ASSERT_EQ(expected_supported_properties, properties.get(ov::supported_properties));

    ASSERT_EQ(0, properties.get(rw_property));
    properties.get(ov::device::properties("sub0", ro_property));
    ASSERT_NO_THROW(properties.get(ov::device::properties("sub1", ro_property)));

    ASSERT_THROW(properties.get(ro_property), ov::Exception);
}
