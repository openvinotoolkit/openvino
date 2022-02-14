// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include <base/ov_behavior_test_utils.hpp>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"

#include <gtest/gtest.h>

using namespace ov::test::behavior;
namespace {

//
// Executable Network GetMetric
//
class OVClassConfigTestCPU : public ::testing::Test,
                             public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
public:
    std::shared_ptr<ngraph::Function> model;
    const std::string deviceName = "CPU";

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        model = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

TEST_F(OVClassConfigTestCPU, smoke_GetROPropertiesDoesNotThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (const auto& property : properties) {
        ASSERT_NO_THROW((void)compiledModel.get_property(property));
    }
}

TEST_F(OVClassConfigTestCPU, smoke_SetROPropertiesThrow) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel compiledModel = ie.compile_model(model, deviceName);

    ASSERT_NO_THROW(properties = compiledModel.get_property(ov::supported_properties));

    for (auto it = properties.begin(); it != properties.end(); ++it) {
        ASSERT_TRUE(it != properties.end());
        ASSERT_FALSE(it->is_mutable());
        ASSERT_THROW(compiledModel.set_property({{*it, "DUMMY VALUE"}}), ov::Exception);
    }
}

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)));

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassExecutableNetworkGetMetricTest,
                         OVClassExecutableNetworkGetMetricTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));

} // namespace
