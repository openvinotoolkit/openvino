// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/auto_unit_test.hpp"
using namespace ov::mock_auto_plugin::tests;

class MultiPropertyTest : public tests::AutoTestWithRealCore, public ::testing::Test {
public:
    void SetUp() override {
        plugin->set_device_name("MULTI");
        std::shared_ptr<ov::IPlugin> base_plugin = plugin;
        reg_plugin(core, base_plugin, "MOCK_MULTI", {});
        // validate mock plugin
        core.get_property("MOCK_MULTI", ov::supported_properties);
    }
};

class AutoPropertyTest : public tests::AutoTestWithRealCore, public ::testing::Test {
public:
    void SetUp() override {
        plugin->set_device_name("AUTO");
        std::shared_ptr<ov::IPlugin> base_plugin = plugin;
        reg_plugin(core, base_plugin, "MOCK_AUTO", {});
        core.get_property("MOCK_AUTO", ov::supported_properties);
    }
};


/* to be enabled if expect multi throw for latency mode
TEST_F(PropertyTest, tputmodeonly_for_multi) {
    EXPECT_THROW_WITH_MESSAGE(core.compile_model(model, "MULTI", ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)), ov::Exception,
                                "MULTI does not support perf mode");
    ASSERT_NO_THROW(compiled_model = core.compile_model(model, "MULTI", ov::device::priorities("MOCK_GPU", "MOCK_CPU")));
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::THROUGHPUT);
}

TEST_F(PropertyTest, tputmodeonly_for_multi_propertyset) {
    ASSERT_NO_THROW(core.get_property("MULTI", ov::supported_properties));
    EXPECT_THROW_WITH_MESSAGE(core.set_property("MULTI", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)), ov::Exception,
                                "MULTI does not support perf mode");
}
*/
/*
TEST_F(PropertyTest, default_perfmode_for_auto) {
    ov::CompiledModel compiled_model;
    EXPECT_NO_THROW(compiled_model = core.compile_model(model, "AUTO", ov::device::priorities("MOCK_GPU", "MOCK_CPU")));
    EXPECT_EQ(compiled_model.get_property(ov::hint::performance_mode), ov::hint::PerformanceMode::LATENCY);
}
*/

TEST_F(MultiPropertyTest, default_perfmode_for_multi) {
    EXPECT_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("THROUGHPUT")))).Times(1);
    EXPECT_CALL(*mock_plugin_gpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("THROUGHPUT")))).Times(1);
    ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, {ov::device::priorities("MOCK_GPU", "MOCK_CPU")}));
    EXPECT_EQ(compiled_model->get_property(ov::hint::performance_mode.name()), ov::hint::PerformanceMode::THROUGHPUT);
}

TEST_F(MultiPropertyTest, respect_secondary_property) {
    EXPECT_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("LATENCY")))).Times(1);
    EXPECT_CALL(*mock_plugin_gpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("LATENCY")))).Times(1);
    ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                    {"DEVICE_PROPERTIES", "{MOCK_CPU:{PERFORMANCE_HINT:LATENCY},MOCK_GPU:{PERFORMANCE_HINT:LATENCY}"}}));
    EXPECT_EQ(compiled_model->get_property(ov::hint::performance_mode.name()), ov::hint::PerformanceMode::THROUGHPUT);
}

TEST_F(AutoPropertyTest, default_perfmode_for_auto_ctput) {
    EXPECT_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("THROUGHPUT")))).Times(1);
    EXPECT_CALL(*mock_plugin_gpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("THROUGHPUT")))).Times(1);
    ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}));
    EXPECT_EQ(compiled_model->get_property(ov::hint::performance_mode.name()), ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
}

TEST_F(AutoPropertyTest, default_perfmode_for_auto) {
    EXPECT_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("LATENCY")))).Times(1);
    EXPECT_CALL(*mock_plugin_gpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("LATENCY")))).Times(1);
    compiled_model = plugin->compile_model(model, {ov::device::priorities("MOCK_GPU", "MOCK_CPU")});
    EXPECT_EQ(compiled_model->get_property(ov::hint::performance_mode.name()), ov::hint::PerformanceMode::LATENCY);
}

TEST_F(AutoPropertyTest, respect_secondary_property_auto_ctput) {
    EXPECT_CALL(*mock_plugin_cpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("LATENCY")))).Times(1);
    EXPECT_CALL(*mock_plugin_gpu.get(), compile_model(::testing::Matcher<const std::shared_ptr<const ov::Model>&>(_),
                ::testing::Matcher<const ov::AnyMap&>(ComparePerfHint("THROUGHPUT")))).Times(1);
    ASSERT_NO_THROW(compiled_model = plugin->compile_model(model, {ov::device::priorities("MOCK_GPU", "MOCK_CPU"),
                    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                    {"DEVICE_PROPERTIES", "{MOCK_CPU:{PERFORMANCE_HINT:LATENCY},MOCK_GPU:{PERFORMANCE_HINT:THROUGHPUT}"}}));
    EXPECT_EQ(compiled_model->get_property(ov::hint::performance_mode.name()), ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT);
}