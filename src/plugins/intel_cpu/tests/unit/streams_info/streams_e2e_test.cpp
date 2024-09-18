// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;

namespace {

struct StreamGenerateionTestCase {
    int input_stream;
    bool input_stream_changed;
    int input_thread;
    int input_request;
    int input_model_prefer;
    int input_socket_id;
    ov::hint::SchedulingCoreType input_type;
    bool input_ht_value;
    bool input_ht_changed;
    bool input_cpu_value;
    bool input_cpu_changed;
    ov::hint::PerformanceMode input_pm_hint;
    std::set<ov::hint::ModelDistributionPolicy> hint_llm_distribution_policy;
    std::vector<std::vector<int>> input_proc_type_table;
    ov::hint::SchedulingCoreType output_type;
    bool output_ht_value;
    bool output_cpu_value;
    ov::hint::PerformanceMode output_pm_hint;
    std::vector<std::vector<int>> output_proc_type_table;
    std::vector<std::vector<int>> output_stream_info_table;
};

void make_config(StreamGenerateionTestCase& test_data, ov::intel_cpu::Config& config) {
    config.schedulingCoreType = test_data.input_type;
    config.enableCpuPinning = test_data.input_cpu_value;
    config.changedCpuPinning = test_data.input_cpu_changed;
    config.enableHyperThreading = test_data.input_ht_value;
    config.changedHyperThreading = test_data.input_ht_changed;
    config.hintPerfMode = test_data.input_pm_hint;
    config.modelDistributionPolicy = test_data.hint_llm_distribution_policy;
    config.hintNumRequests = test_data.input_request;
    config.streams = test_data.input_stream_changed ? test_data.input_stream
                                                    : (test_data.input_stream == 0 ? 1 : test_data.input_stream);
    config.streamsChanged = test_data.input_stream_changed;
    config.threads = test_data.input_thread;
}

class StreamGenerationTests : public ov::test::TestsCommon,
                              public testing::WithParamInterface<std::tuple<StreamGenerateionTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());
        ov::intel_cpu::Config config;
        make_config(test_data, config);

        CPU& cpu = cpu_info();
        cpu._proc_type_table = test_data.input_proc_type_table;

        auto proc_type_table = ov::intel_cpu::generate_stream_info(test_data.input_stream,
                                                                   test_data.input_socket_id,
                                                                   nullptr,
                                                                   config,
                                                                   test_data.input_proc_type_table,
                                                                   test_data.input_model_prefer);

        ASSERT_EQ(test_data.output_stream_info_table, config.streamExecutorConfig.get_streams_info_table());
        ASSERT_EQ(test_data.output_proc_type_table, proc_type_table);
        ASSERT_EQ(test_data.output_cpu_value, config.streamExecutorConfig.get_cpu_pinning());
        ASSERT_EQ(test_data.output_ht_value, config.enableHyperThreading);
        ASSERT_EQ(test_data.output_type, config.schedulingCoreType);
        ASSERT_EQ(test_data.output_pm_hint, config.hintPerfMode);
    }
};

TEST_P(StreamGenerationTests, StreamsGeneration) {}

StreamGenerateionTestCase generation_latency_1sockets_14cores_1_pinning = {
    1,                                       // param[in]: simulated settting for streams number
    false,                                   // param[in]: simulated settting for streams number changed
    0,                                       // param[in]: simulated setting for threads number
    0,                                       // param[in]: simulated setting for inference request number
    0,                                       // param[in]: simulated setting for model prefer threads number
    0,                                       // param[in]: simulated setting for socket id of running thread
    ov::hint::SchedulingCoreType::ANY_CORE,  // param[in]: simulated setting for scheduling core type
                                             // (PCORE_ONLY/ECORE_ONLY/ANY_CORE)
    true,                                    // param[in]: simulated setting for enableHyperThreading
    true,                                    // param[in]: simulated settting for changedHyperThreading
    true,                                    // param[in]: simulated setting for enableCpuPinning
    true,                                    // param[in]: simulated setting for changedCpuPinning
    ov::hint::PerformanceMode::LATENCY,      // param[in]: simulated setting for performance mode (throughput/latency)
    {},  // param[in]: simulated setting for model distribution policy
    {{20, 6, 8, 6, 0, 0}},  // param[in]: simulated proc_type_table for platform which has one socket, 6 Pcores, 8
                            // Ecores and hyper threading enabled
    ov::hint::SchedulingCoreType::ANY_CORE,  // param[expected out]: scheduling core type needs to be the same as input
    true,                                    // param[expected out]: enableHyperThreading needs to be the same as input
    true,                                    // param[expected out]: enableCpuPinning needs to be the same as input
    ov::hint::PerformanceMode::LATENCY,      // param[expected out]: performance mode needs to be the same as input
    {{20, 6, 8, 6, 0, 0}},  // param[expected out]: since hyper threading is enabled and all core type is used,
                            // proc_type_table needs to be the same as input
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},  // param[expected out]: since performance mode is latency and all cores is
                                           // used, the final streams is 1
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_2_pinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{14, 6, 8, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_1_pinning = {
    0,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, EFFICIENT_CORE_PROC, 3, 0, 0}, {2, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_1_unpinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,  // param[expected out]: enableCpuPinning needs to be false becuase OS cannot support thread pinning
    ov::hint::PerformanceMode::LATENCY,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_2_unpinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{14, 6, 8, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_1_unpinning = {
    0,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, EFFICIENT_CORE_PROC, 3, 0, 0}, {2, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_3 = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{14, 6, 8, 0, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_4 = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{12, 6, 0, 6, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamGenerateionTestCase generation_latency_1sockets_14cores_5 = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_6 = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_7 = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_8 = {
    1,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_9 = {
    1,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_10 = {
    1,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 48, -1, -1}, {-1, MAIN_CORE_PROC, 24, 0, 0}, {-1, MAIN_CORE_PROC, 24, 1, 1}},
};

StreamGenerateionTestCase generation_latency_2sockets_48cores_11 = {
    1,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL},
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 48, -1, -1}, {-1, MAIN_CORE_PROC, 24, 0, 0}, {-1, MAIN_CORE_PROC, 24, 1, 1}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_2 = {
    0,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{6, 6, 0, 0, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_3 = {
    10,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{12, 6, 0, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {4, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamGenerateionTestCase generation_tput_1sockets_14cores_4 = {
    0,
    false,
    10,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{20, 6, 8, 6, 0, 0}},
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {1, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_5 = {
    0,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    {{6, MAIN_CORE_PROC, 4, 0, 0},
     {6, MAIN_CORE_PROC, 4, 1, 1},
     {6, HYPER_THREADING_PROC, 4, 0, 0},
     {6, HYPER_THREADING_PROC, 4, 1, 1}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_6 = {
    0,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{6, MAIN_CORE_PROC, 4, 0, 0}, {6, MAIN_CORE_PROC, 4, 1, 1}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_7 = {
    100,
    true,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{24, MAIN_CORE_PROC, 1, 0, 0}, {24, MAIN_CORE_PROC, 1, 1, 1}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_8 = {
    2,
    true,
    20,
    0,
    1,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 10, 0, 0}},
};

StreamGenerateionTestCase generation_tput_2sockets_48cores_9 = {
    0,
    false,
    0,
    0,
    1,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    false,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 48, 0, 48, -1, -1}, {48, 24, 0, 24, 0, 0}, {48, 24, 0, 24, 1, 1}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{24, MAIN_CORE_PROC, 1, 0, 0}, {24, MAIN_CORE_PROC, 1, 1, 1}},
};
StreamGenerateionTestCase generation_latency_1sockets_96cores_pinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {{96, 0, 96, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 96, 0, 0}},
};
StreamGenerateionTestCase generation_tput_1sockets_96cores_pinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 0, 96, 0, 0, 0}},
    {{24, EFFICIENT_CORE_PROC, 4, 0, 0}},
};
StreamGenerateionTestCase generation_tput_1sockets_96cores_2_pinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    true,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 0, 96, 0, 0, 0}},
    {{24, EFFICIENT_CORE_PROC, 4, 0, 0}},
};
StreamGenerateionTestCase generation_latency_1sockets_96cores_unpinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    true,
    true,
    ov::hint::PerformanceMode::LATENCY,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::LATENCY,
    {{96, 0, 96, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 96, 0, 0}},
};
StreamGenerateionTestCase generation_tput_1sockets_96cores_unpinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 0, 96, 0, 0, 0}},
    {{24, EFFICIENT_CORE_PROC, 4, 0, 0}},
};
StreamGenerateionTestCase generation_tput_1sockets_96cores_2_unpinning = {
    1,
    false,
    0,
    0,
    0,
    0,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    true,
    false,
    true,
    ov::hint::PerformanceMode::THROUGHPUT,
    {},
    {{96, 0, 96, 0, 0, 0}},
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    false,
    ov::hint::PerformanceMode::THROUGHPUT,
    {{96, 0, 96, 0, 0, 0}},
    {{24, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

#if defined(__linux__) || defined(_WIN32)
INSTANTIATE_TEST_SUITE_P(smoke_StreamsGeneration,
                         StreamGenerationTests,
                         ::testing::Values(generation_latency_1sockets_14cores_3,
                                           generation_latency_1sockets_14cores_4,
                                           generation_latency_1sockets_14cores_5,
                                           generation_latency_2sockets_48cores_6,
                                           generation_latency_2sockets_48cores_7,
                                           generation_latency_2sockets_48cores_8,
                                           generation_latency_2sockets_48cores_9,
                                           generation_latency_2sockets_48cores_10,
                                           generation_latency_2sockets_48cores_11,
                                           generation_latency_1sockets_14cores_1_pinning,
                                           generation_latency_1sockets_14cores_2_pinning,
                                           generation_tput_1sockets_14cores_1_pinning,
                                           generation_tput_1sockets_14cores_2,
                                           generation_tput_1sockets_14cores_3,
                                           generation_tput_1sockets_14cores_4,
                                           generation_tput_2sockets_48cores_5,
                                           generation_tput_2sockets_48cores_6,
                                           generation_tput_2sockets_48cores_7,
                                           generation_tput_2sockets_48cores_8,
                                           generation_tput_2sockets_48cores_9,
                                           generation_latency_1sockets_96cores_pinning,
                                           generation_tput_1sockets_96cores_pinning,
                                           generation_tput_1sockets_96cores_2_pinning));
#else
INSTANTIATE_TEST_SUITE_P(smoke_StreamsGeneration,
                         StreamGenerationTests,
                         ::testing::Values(generation_latency_1sockets_14cores_3,
                                           generation_latency_1sockets_14cores_4,
                                           generation_latency_1sockets_14cores_5,
                                           generation_latency_2sockets_48cores_6,
                                           generation_latency_2sockets_48cores_7,
                                           generation_latency_2sockets_48cores_8,
                                           generation_latency_2sockets_48cores_9,
                                           generation_latency_2sockets_48cores_10,
                                           generation_latency_2sockets_48cores_11,
                                           generation_latency_1sockets_14cores_1_unpinning,
                                           generation_latency_1sockets_14cores_2_unpinning,
                                           generation_tput_1sockets_14cores_1_unpinning,
                                           generation_tput_1sockets_14cores_2,
                                           generation_tput_1sockets_14cores_3,
                                           generation_tput_1sockets_14cores_4,
                                           generation_tput_2sockets_48cores_5,
                                           generation_tput_2sockets_48cores_6,
                                           generation_tput_2sockets_48cores_7,
                                           generation_tput_2sockets_48cores_8,
                                           generation_tput_2sockets_48cores_9,
                                           generation_latency_1sockets_96cores_unpinning,
                                           generation_tput_1sockets_96cores_unpinning,
                                           generation_tput_1sockets_96cores_2_unpinning));

#endif
}  // namespace
