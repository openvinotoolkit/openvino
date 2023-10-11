// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "hetero_tests.hpp"
#include "ie/ie_plugin_config.hpp"
#include "openvino/runtime/internal_properties.hpp"

using namespace ov::hetero::tests;

TEST_F(HeteroTests, get_property_supported_properties) {
    const std::vector<ov::PropertyName> supported_properties = {ov::supported_properties,
                                                                ov::device::full_name,
                                                                ov::device::capabilities,
                                                                ov::device::priorities};
    auto actual_supported_properties = core.get_property("HETERO", ov::supported_properties);
    EXPECT_EQ(supported_properties.size(), actual_supported_properties.size());
    for (auto& supported_property : supported_properties) {
        ASSERT_TRUE(std::find(actual_supported_properties.begin(),
                              actual_supported_properties.end(),
                              supported_property) != actual_supported_properties.end());
    }
}

TEST_F(HeteroTests, get_property_supported_metrics) {
    const std::vector<std::string> supported_metrics = {ov::supported_properties.name(),
                                                        ov::device::full_name.name(),
                                                        ov::device::capabilities.name(),
                                                        METRIC_KEY(SUPPORTED_METRICS),
                                                        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                        METRIC_KEY(IMPORT_EXPORT_SUPPORT)};
    auto actual_supported_metrics =
        core.get_property("HETERO", METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
    EXPECT_EQ(supported_metrics.size(), actual_supported_metrics.size());
    for (auto& supported_metric : supported_metrics) {
        ASSERT_TRUE(std::find(actual_supported_metrics.begin(), actual_supported_metrics.end(), supported_metric) !=
                    actual_supported_metrics.end());
    }
}

TEST_F(HeteroTests, get_property_supported_configs) {
    const std::vector<std::string> supported_configs = {"HETERO_DUMP_GRAPH_DOT",
                                                        "TARGET_FALLBACK",
                                                        ov::device::priorities.name()};
    auto actual_supported_configs =
        core.get_property("HETERO", METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
    EXPECT_EQ(supported_configs.size(), actual_supported_configs.size());
    for (auto& supported_config : supported_configs) {
        ASSERT_TRUE(std::find(actual_supported_configs.begin(), actual_supported_configs.end(), supported_config) !=
                    actual_supported_configs.end());
    }
}

TEST_F(HeteroTests, get_property_internal_supported_properties) {
    const std::vector<ov::PropertyName> supported_properties = {ov::internal::caching_properties};
    auto actual_supported_properties = core.get_property("HETERO", ov::internal::supported_properties);
    EXPECT_EQ(supported_properties.size(), actual_supported_properties.size());
    for (auto& supported_property : supported_properties) {
        ASSERT_TRUE(std::find(actual_supported_properties.begin(),
                              actual_supported_properties.end(),
                              supported_property) != actual_supported_properties.end());
    }
}

TEST_F(HeteroTests, get_property_ro_properties) {
    EXPECT_EQ("HETERO", core.get_property("HETERO", ov::device::full_name));
    EXPECT_EQ(std::vector<std::string>{ov::device::capability::EXPORT_IMPORT},
              core.get_property("HETERO", ov::device::capabilities));
}

TEST_F(HeteroTests, set_property_device_priorities) {
    EXPECT_EQ("", core.get_property("HETERO", ov::device::priorities));
    core.set_property("HETERO", ov::device::priorities("MOCK0,MOCK1"));
    EXPECT_EQ("MOCK0,MOCK1", core.get_property("HETERO", ov::device::priorities));
    EXPECT_EQ("MOCK0,MOCK1", core.get_property("HETERO", "TARGET_FALLBACK").as<std::string>());
    core.set_property("HETERO", {{"TARGET_FALLBACK", "MOCK1,MOCK0"}});
    EXPECT_EQ("MOCK1,MOCK0", core.get_property("HETERO", ov::device::priorities));
    EXPECT_EQ("MOCK1,MOCK0", core.get_property("HETERO", "TARGET_FALLBACK").as<std::string>());
}