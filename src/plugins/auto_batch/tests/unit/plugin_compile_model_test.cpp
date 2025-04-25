// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/conv_pool_relu_non_zero.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"
#include "mock_common.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

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
            .WillByDefault(Return(m_core_properities[ov::hint::performance_mode.name()]));

        ON_CALL(*m_core, get_property(_, StrEq("OPTIMAL_BATCH_SIZE"), _))
            .WillByDefault(Return(m_core_properities[ov::optimal_batch_size.name()]));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
            .WillByDefault(Return(m_core_properities[ov::hint::num_requests.name()]));

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
            .WillByDefault(Return(m_core_properities[ov::intel_gpu::device_total_mem_size.name()]));

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
    m_model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities));
}

TEST_P(PluginCompileModelTest, PluginCompileModelWithRemoteContextTestCase) {
    m_model = ov::test::utils::make_multi_single_conv();
    OV_ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities, m_remote_context));
}

TEST_P(PluginCompileModelTest, PluginCompileModelBatchedModelTestCase) {
    m_model = ov::test::utils::make_conv_pool_relu_non_zero({1, 1, 32, 32});
    auto batch = ov::Dimension(5);
    batch.set_symbol(std::make_shared<ov::Symbol>());
    auto p_shape = ov::PartialShape{batch, 1, 32, 32};
    m_model->reshape(p_shape);
    OV_ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities));
}

TEST_P(PluginCompileModelTest, PluginCompileModelBatchedModelWithRemoteContextTestCase) {
    m_model = ov::test::utils::make_conv_pool_relu_non_zero({1, 1, 32, 32});
    auto batch = ov::Dimension(5);
    batch.set_symbol(std::make_shared<ov::Symbol>());
    auto p_shape = ov::PartialShape{batch, 1, 32, 32};
    m_model->reshape(p_shape);
    OV_ASSERT_NO_THROW(m_plugin->compile_model(m_model, m_plugin_properities, m_remote_context));
}

const std::vector<plugin_compile_model_param> plugin_compile_model_param_test = {
    // Case 1: explicitly apply batch size by config of AUTO_BATCH_DEVICE_CONFIG
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU(32)")}},
                               32},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("GPU(32)")}},
                               32},
    // Case 2: CPU batch size is figured out by min of opt_batch_size and infReq_num
    //         If config contains "PERFORMANCE_HINT_NUM_REQUESTS"
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU")}},
                               12},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(8)},
                                {ov::hint::num_requests(16)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU")}},
                               8},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(8)},
                                {ov::hint::num_requests(2)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU")}},
                               1},
    // Case 3: GPU batch size is figured out by
    //      1) min of opt_batch_size and infReq_num
    //      2) available_mem/one_graph_mem_footprint with power 2
    //  Final m_batch_size is the min of 1) and 2)
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(5000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("GPU")}},
                               4},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(40960000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("GPU")}},
                               12},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(32)},
                                {ov::hint::num_requests(24)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(18000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("GPU")}},
                               16},
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(32)},
                                {ov::hint::num_requests(48)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(180000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("GPU")}},
                               32},
    // Case 4:
    plugin_compile_model_param{{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
                                {ov::optimal_batch_size.name(), static_cast<unsigned int>(16)},
                                {ov::hint::num_requests(12)},
                                {ov::intel_gpu::memory_statistics.name(), static_cast<uint64_t>(1024000)},
                                {ov::intel_gpu::device_total_mem_size.name(), static_cast<uint64_t>(4096000000)}},
                               {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU(32)")}},
                               32},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         PluginCompileModelTest,
                         ::testing::ValuesIn(plugin_compile_model_param_test),
                         PluginCompileModelTest::getTestCaseName);
