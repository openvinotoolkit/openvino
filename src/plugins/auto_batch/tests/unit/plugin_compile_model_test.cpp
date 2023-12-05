// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;

using namespace ov::mock_autobatch_plugin;

using plugin_compile_model_param = std::tuple<ov::AnyMap,  // Core Properties
                                              ov::AnyMap,  // Plugin Properties
                                              uint32_t>;   // batch size

class PluginCompileModelTest : public ::testing::TestWithParam<plugin_compile_model_param> {
public:
    ov::AnyMap m_core_properities;
    ov::AnyMap m_plugin_properities;
    int m_batch_size;

    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    std::shared_ptr<ov::Model> m_model;
    ov::SoPtr<ov::IRemoteContext> m_remote_context;

    ov::SoPtr<MockICompiledModel> m_mock_compile_model;
    std::shared_ptr<MockICompiledModel> m_mock_i_compile_model;
    std::shared_ptr<NiceMock<MockIPlugin>> m_hardware_plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<plugin_compile_model_param> obj) {
        ov::AnyMap core_properities;
        ov::AnyMap plugin_properities;
        uint32_t expect_batch_size;
        std::tie(core_properities, plugin_properities, expect_batch_size) = obj.param;

        std::string res;
        for (auto& c : core_properities) {
            res += "_" + c.first + "_" + c.second.as<std::string>();
        }
        for (auto& c : plugin_properities) {
            res += "_" + c.first + "_" + c.second.as<std::string>();
        }
        res += "_" + std::to_string(expect_batch_size);
        return res;
    }

    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
        m_model.reset();
        m_remote_context = {};
        m_mock_i_compile_model.reset();
        m_mock_compile_model = {};
    }

    void SetUp() override {
        std::tie(m_core_properities, m_plugin_properities, m_batch_size) = this->GetParam();
        m_core = std::shared_ptr<NiceMock<ov::MockICore>>(new NiceMock<ov::MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);
        m_hardware_plugin = std::shared_ptr<NiceMock<MockIPlugin>>(new NiceMock<MockIPlugin>());
        m_mock_i_compile_model = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_hardware_plugin);
        m_mock_compile_model = {m_mock_i_compile_model, {}};

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT")))
            .WillByDefault(Return(m_core_properities["PERFORMANCE_HINT"]));

        ON_CALL(*m_core, get_property(_, StrEq("OPTIMAL_BATCH_SIZE"), _))
            .WillByDefault(Return(m_core_properities["OPTIMAL_BATCH_SIZE"]));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
            .WillByDefault(Return(m_core_properities["PERFORMANCE_HINT_NUM_REQUESTS"]));

        ON_CALL(*m_core, get_property(_, StrEq("GPU_MEMORY_STATISTICS"), _))
            .WillByDefault([&](const std::string& device, const std::string& key, const ov::AnyMap& options) {
                static int flag = 0;
                ov::Any value = m_core_properities[key];
                uint64_t data = flag * value.as<uint64_t>();
                std::map<std::string, uint64_t> ret = {{"xyz", data}};
                flag = flag ? 0 : 1;
                return ret;
            });

        ON_CALL(*m_core, get_property(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _))
            .WillByDefault(Return(m_core_properities["GPU_DEVICE_TOTAL_MEM_SIZE"]));

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const std::string&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const ov::SoPtr<ov::IRemoteContext>&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));
    }
};

TEST_P(PluginCompileModelTest, PluginCompileModelTestCase) {
    m_model = ngraph::builder::subgraph::makeMultiSingleConv();
    ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities));
}

TEST_P(PluginCompileModelTest, PluginCompileModelWithRemoteContextTestCase) {
    m_model = ngraph::builder::subgraph::makeMultiSingleConv();
    ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities, m_remote_context));
}

TEST_P(PluginCompileModelTest, PluginCompileModelBatchedModelTestCase) {
    m_model = ngraph::builder::subgraph::makeConvPoolReluNonZero({1, 1, 32, 32});
    auto batch = ov::Dimension(5);
    ov::DimensionTracker::set_label(batch, 11);
    auto p_shape = ov::PartialShape{batch, 1, 32, 32};
    m_model->reshape(p_shape);
    ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities));
}

TEST_P(PluginCompileModelTest, PluginCompileModelBatchedModelWithRemoteContextTestCase) {
    m_model = ngraph::builder::subgraph::makeConvPoolReluNonZero({1, 1, 32, 32});
    auto batch = ov::Dimension(5);
    ov::DimensionTracker::set_label(batch, 11);
    auto p_shape = ov::PartialShape{batch, 1, 32, 32};
    m_model->reshape(p_shape);
    ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities, m_remote_context));
}

const std::vector<plugin_compile_model_param> plugin_compile_model_param_test = {
    // Case 1: explict apply batch size by config of AUTO_BATCH_DEVICE_CONFIG
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                               32},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU(32)"}},
                               32},
    // Case 2: CPU batch size is figured out by min of opt_batch_size and infReq_num
    //         If config contains "PERFORMANCE_HINT_NUM_REQUESTS"
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                               12},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(8)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(16)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                               8},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(8)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(2)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU"}},
                               1},
    // Case 3: GPU batch size is figured out by
    //      1) min of opt_batch_size and infReq_num
    //      2) available_mem/one_graph_mem_footprint with power 2
    //  Final m_batch_size is the min of 1) and 2)
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "5000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                               4},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "40960000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                               12},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(32)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(24)},
                                {"GPU_MEMORY_STATISTICS", "1000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "18000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                               16},
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::THROUGHPUT},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(32)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(48)},
                                {"GPU_MEMORY_STATISTICS", "1000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "180000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "GPU"}},
                               32},
    // Case 4:
    plugin_compile_model_param{{{"PERFORMANCE_HINT", ov::hint::PerformanceMode::LATENCY},
                                {"OPTIMAL_BATCH_SIZE", static_cast<unsigned int>(16)},
                                {"PERFORMANCE_HINT_NUM_REQUESTS", static_cast<uint32_t>(12)},
                                {"GPU_MEMORY_STATISTICS", "1024000"},
                                {"GPU_DEVICE_TOTAL_MEM_SIZE", "4096000000"}},
                               {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(32)"}},
                               32},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         PluginCompileModelTest,
                         ::testing::ValuesIn(plugin_compile_model_param_test),
                         PluginCompileModelTest::getTestCaseName);
