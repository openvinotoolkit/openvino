// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common_test_utils/test_constants.hpp"
#include "hetero_tests.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::hetero::tests;

TEST_F(HeteroTests, get_available_devices) {
    auto available_devices = core.get_available_devices();
    std::vector<std::string> mock_reference_dev = {{"MOCK0.0"}, {"MOCK0.1"}, {"MOCK0.2"}, {"MOCK1.0"}, {"MOCK1.1"}};
    for (const auto& dev : available_devices) {
        auto it = std::find(mock_reference_dev.begin(), mock_reference_dev.end(), dev);
        if (it != mock_reference_dev.end()) {
            mock_reference_dev.erase(it);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}

TEST_F(HeteroTests, compile_with_registered_devices) {
    // Change device priority
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
    auto model = create_model_with_reshape();
    EXPECT_NO_THROW(core.compile_model(model, ov::test::utils::DEVICE_HETERO));
}

TEST_F(HeteroTests, compile_with_unregistered_devices_throw) {
    // Change device priority
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK2,MOCK3"));
    auto model = create_model_with_reshape();
    EXPECT_THROW(core.compile_model(model, ov::test::utils::DEVICE_HETERO), ov::Exception);
}

TEST_F(HeteroTests, compile_without_device_priorities_throw) {
    // Change device priority
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities(""));
    auto model = create_model_with_reshape();
    EXPECT_THROW(core.compile_model(model, ov::test::utils::DEVICE_HETERO), ov::Exception);
}

TEST_F(HeteroTests, compile_dynamic_model_fail) {
    // Change device priority
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
    auto model = create_model_with_subtract_reshape(true);
    EXPECT_THROW(core.compile_model(model, ov::test::utils::DEVICE_HETERO), ov::Exception);
}

TEST_F(HeteroTests, compile_model_shapeof) {
    // Change device priority
    core.set_property(ov::test::utils::DEVICE_HETERO, ov::device::priorities("MOCK0,MOCK1"));
    auto model = create_model_with_subtract_shapeof_reshape();
    EXPECT_NO_THROW(core.compile_model(model, ov::test::utils::DEVICE_HETERO));
}

TEST_F(HeteroTests, compile_with_device_properties) {
    ov::AnyMap config = {ov::device::priorities("MOCK0,MOCK1"),
                         ov::device::properties("MOCK0", ov::num_streams(4), ov::enable_profiling(false)),
                         ov::device::properties("MOCK1", ov::num_streams(6), ov::enable_profiling(true))};
    auto model = create_model_with_subtract_reshape();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, config);
    EXPECT_THROW(compiled_model.get_property(ov::num_streams), ov::Exception);
    EXPECT_THROW(compiled_model.get_property(ov::enable_profiling), ov::Exception);
    auto device_properties = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    ASSERT_TRUE(device_properties.count("MOCK0.0"));
    auto mock0_properties = device_properties.at("MOCK0.0").as<ov::AnyMap>();
    ASSERT_TRUE(mock0_properties.count(ov::num_streams.name()));
    ASSERT_TRUE(mock0_properties.count(ov::enable_profiling.name()));
    EXPECT_EQ(1, mock0_properties.at(ov::num_streams.name()).as<ov::streams::Num>());
    EXPECT_EQ(false, mock0_properties.at(ov::enable_profiling.name()).as<bool>());
    ASSERT_TRUE(device_properties.count("MOCK1.0"));
    auto mock1_properties = device_properties.at("MOCK1.0").as<ov::AnyMap>();
    ASSERT_TRUE(mock1_properties.count(ov::num_streams.name()));
    ASSERT_TRUE(mock1_properties.count(ov::enable_profiling.name()));
    EXPECT_EQ(6, mock1_properties.at(ov::num_streams.name()).as<ov::streams::Num>());
    EXPECT_EQ(true, mock1_properties.at(ov::enable_profiling.name()).as<bool>());
}

TEST_F(HeteroTests, compile_with_device_properties_no_exclusive) {
    ov::AnyMap config = {ov::device::priorities("MOCK0,MOCK1"),
                         ov::internal::exclusive_async_requests(false),
                         ov::device::properties("MOCK0", ov::num_streams(4)),
                         ov::device::properties("MOCK1", ov::num_streams(6))};
    auto model = create_model_with_subtract_reshape();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, config);
    EXPECT_THROW(compiled_model.get_property(ov::num_streams), ov::Exception);
    auto device_properties = compiled_model.get_property(ov::device::properties.name()).as<ov::AnyMap>();
    ASSERT_TRUE(device_properties.count("MOCK0.0"));
    auto mock0_properties = device_properties.at("MOCK0.0").as<ov::AnyMap>();
    ASSERT_TRUE(mock0_properties.count(ov::num_streams.name()));
    EXPECT_EQ(4, mock0_properties.at(ov::num_streams.name()).as<ov::streams::Num>());
    ASSERT_TRUE(device_properties.count("MOCK1.0"));
    auto mock1_properties = device_properties.at("MOCK1.0").as<ov::AnyMap>();
    ASSERT_TRUE(mock1_properties.count(ov::num_streams.name()));
    EXPECT_EQ(6, mock1_properties.at(ov::num_streams.name()).as<ov::streams::Num>());
}

TEST_F(HeteroTests, get_runtime_model) {
    ov::AnyMap config = {ov::device::priorities("MOCK0,MOCK1")};
    auto model = create_model_with_subtract_reshape();
    std::set<std::string> original_names;
    for (auto& op : model->get_ordered_ops()) {
        original_names.insert(op->get_friendly_name());
    }
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_HETERO, config);
    auto runtime_model = compiled_model.get_runtime_model();
    for (auto& op : runtime_model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        ASSERT_TRUE(info.count(ov::exec_model_info::EXECUTION_ORDER));
        ASSERT_TRUE(info.count(ov::exec_model_info::IMPL_TYPE));
        ASSERT_TRUE(info.count(ov::exec_model_info::PERF_COUNTER));
        ASSERT_TRUE(info.count(ov::exec_model_info::ORIGINAL_NAMES));
        auto fused_names = info.at(ov::exec_model_info::ORIGINAL_NAMES).as<std::vector<std::string>>();
        for (auto& fused_name : fused_names) {
            if (original_names.count(fused_name))
                original_names.erase(fused_name);
        }
        ASSERT_TRUE(info.count(ov::exec_model_info::RUNTIME_PRECISION));
        ASSERT_TRUE(info.count(ov::exec_model_info::OUTPUT_PRECISIONS));
    }
    EXPECT_EQ(0, original_names.size());
}