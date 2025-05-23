// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "properties.hpp"

namespace ov {
namespace hetero {
namespace tests {

TEST_F(HeteroTests, get_property_supported_properties) {
    const std::vector<ov::PropertyName> supported_properties = {ov::supported_properties,
                                                                ov::device::full_name,
                                                                ov::device::capabilities,
                                                                ov::device::priorities,
                                                                ov::hint::model_distribution_policy};
    auto actual_supported_properties = core.get_property(ov::test::utils::DEVICE_HETERO, ov::supported_properties);
    EXPECT_EQ(supported_properties.size(), actual_supported_properties.size());
    for (auto& supported_property : supported_properties) {
        ASSERT_TRUE(std::find(actual_supported_properties.begin(),
                              actual_supported_properties.end(),
                              supported_property) != actual_supported_properties.end());
    }
}

TEST_F(HeteroTests, get_property_internal_supported_properties) {
    const std::vector<ov::PropertyName> supported_properties = {ov::internal::caching_properties};
    auto actual_supported_properties =
        core.get_property(ov::test::utils::DEVICE_HETERO, ov::internal::supported_properties);
    EXPECT_EQ(supported_properties.size(), actual_supported_properties.size());
    for (auto& supported_property : supported_properties) {
        ASSERT_TRUE(std::find(actual_supported_properties.begin(),
                              actual_supported_properties.end(),
                              supported_property) != actual_supported_properties.end());
    }
}

TEST_F(HeteroTests, get_property_ro_properties) {
    EXPECT_EQ(ov::test::utils::DEVICE_HETERO, core.get_property(ov::test::utils::DEVICE_HETERO, ov::device::full_name));
    EXPECT_EQ(std::vector<std::string>{ov::device::capability::EXPORT_IMPORT},
              core.get_property(ov::test::utils::DEVICE_HETERO, ov::device::capabilities));
}

TEST_F(HeteroTests, set_property_device_priorities) {
    EXPECT_EQ("", core.get_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities));
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
    EXPECT_EQ("MOCK0,MOCK1", core.get_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities));
}

TEST_F(HeteroTests, set_property_ModelDistributionPolicy) {
    std::set<ov::hint::ModelDistributionPolicy> value = {};
    std::set<ov::hint::ModelDistributionPolicy> model_policy = {ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL};

    ASSERT_NO_THROW(
        core.set_property(ov::test::utils::DEVICE_HETERO, ov::hint::model_distribution_policy(model_policy)));
    ASSERT_NO_THROW(value = core.get_property(ov::test::utils::DEVICE_HETERO, ov::hint::model_distribution_policy));
    ASSERT_EQ(model_policy, value);

    model_policy = {};

    ASSERT_NO_THROW(
        core.set_property(ov::test::utils::DEVICE_HETERO, ov::hint::model_distribution_policy(model_policy)));
    ASSERT_NO_THROW(value = core.get_property(ov::test::utils::DEVICE_HETERO, ov::hint::model_distribution_policy));
    ASSERT_EQ(model_policy, value);
}
}  // namespace tests
}  // namespace hetero
}  // namespace ov