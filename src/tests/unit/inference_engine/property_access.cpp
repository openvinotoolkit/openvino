// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gtest/gtest.h>

#include <properties.hpp>
#include <openvino/runtime/properties.hpp>
#include "ie_plugin_config.hpp"


ov::Property<int, ov::PropertyMutability::RO> ro_property{"RO_PROPERTY"};
ov::Property<int, ov::PropertyMutability::RW> rw_property{"RW_PROPERTY"};
ov::Property<std::string, ov::PropertyMutability::RW> str_property{"RW_PROPERTY"};

TEST(OVPropertyAccessTests, empty) {
    ov::PropertyAccess properties;
    ASSERT_TRUE(properties.empty());
}

TEST(OVPropertyAccessTests, bindROConstant) {
    ov::PropertyAccess properties;
    properties.add(ro_property, 42);
    ASSERT_EQ(42, properties.get(ro_property));
    ASSERT_THROW(properties.set(ro_property.name(), 0), ov::Exception);
}

TEST(OVPropertyAccessTests, bindRWCStr) {
    ov::PropertyAccess properties;
    properties.add("rw_property", "42");
    ASSERT_EQ(std::string{"42"}, properties.get("rw_property").as<std::string>());
    ASSERT_NO_THROW(properties.set("rw_property", ""));
}

TEST(OVPropertyAccessTests, bindRWStr) {
    ov::PropertyAccess properties;
    properties.add(str_property, "42");
    ASSERT_EQ(std::string{"42"}, properties.get(str_property));
    ASSERT_THROW(properties.set("ro_property", ""), ov::Exception);
}

TEST(OVPropertyAccessTests, bindROtoRWProperty) {
    ov::PropertyAccess properties;
    properties.add(rw_property, 42, ov::PropertyMutability::RO);
    ASSERT_EQ(42, properties.get(rw_property));
    ASSERT_THROW(properties.set(rw_property(0)), ov::Exception);
}

TEST(OVPropertyAccessTests, bindROGetter) {
    ov::PropertyAccess properties;
    properties.add(ro_property, [] {return 42;});
    ASSERT_EQ(42, properties.get(ro_property));
    ASSERT_THROW(properties.set(ov::AnyMap{{ro_property.name(), 0}}), ov::Exception);
}

TEST(OVPropertyAccessTests, bindRWGetterSetter) {
    ov::PropertyAccess properties;
    int v = 0;
    properties.add(rw_property, [&] {return v;}, [&] (const int& x) {v = 42;});
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertyAccessTests, bindRWRef) {
    ov::PropertyAccess properties;
    int v = 0;
    properties.add(rw_property, std::ref(v));
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertyAccessTests, bindRWStoredInternally) {
    ov::PropertyAccess properties;
    properties.add(rw_property, 0);
    ASSERT_EQ(0, properties.get(rw_property));
    properties.set(rw_property(42));
    ASSERT_EQ(42, properties.get(rw_property));
}

TEST(OVPropertyAccessTests, precondition) {
    ov::PropertyAccess properties;
    properties.add(rw_property, 42, [] (const int& v) {OPENVINO_ASSERT(v < 42);});
    ASSERT_EQ(42, properties.get(rw_property));
    ASSERT_NO_THROW(properties.set(rw_property(41)));
    ASSERT_THROW(properties.set(rw_property(43)), ov::Exception);
    ASSERT_EQ(41, properties.get(rw_property));
}

TEST(OVPropertyAccessTests, bindSubProperties) {
    ov::PropertyAccess properties;
    ov::PropertyAccess sub_properties;
    sub_properties.add(ro_property, 42);
    sub_properties.add(rw_property, 42);
    properties.add("sub", sub_properties);

    auto expected_supported_properties = std::vector<ov::PropertyName>{
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(SUPPORTED_METRICS),
        ov::supported_properties.name(),
        ro_property.name(),     // NOTE: AS this property name is unique
        rw_property.name()};    // it is represented with redused path thus it is situated in sub property set

    ASSERT_EQ(expected_supported_properties, properties.get(ov::supported_properties));

    ASSERT_EQ(42, properties.get(ov::properties("sub", ro_property)));
    properties.set(ov::properties("sub", rw_property(24)));
    ASSERT_EQ(24, properties.get(ov::properties("sub", rw_property)));
}

TEST(OVPropertyAccessTests, ambiguousNameThrow) {
    ov::PropertyAccess properties;
    properties.add(rw_property, 0);
    {
        ov::PropertyAccess sub_properties;
        sub_properties.add(ro_property, 42);
        properties.add("sub0", sub_properties);
    }
    {
        ov::PropertyAccess sub_properties;
        sub_properties.add(ro_property, 42);
        properties.add("sub1", sub_properties);
    }
    auto expected_supported_properties = std::vector<ov::PropertyName>{
        rw_property.name(),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(SUPPORTED_METRICS),
        ov::supported_properties.name(),
        std::string {"sub0."} + ro_property.name(),
        std::string {"sub1."} + ro_property.name()};

    ASSERT_EQ(expected_supported_properties, properties.get(ov::supported_properties));

    ASSERT_EQ(0, properties.get(rw_property));
    properties.get(ov::properties("sub0", ro_property));
    ASSERT_NO_THROW(properties.get(ov::properties("sub1", ro_property)));

    ASSERT_THROW(properties.get(ro_property), ov::Exception);
}
