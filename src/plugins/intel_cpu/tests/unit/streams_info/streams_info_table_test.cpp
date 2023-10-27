// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_system_conf.h>

#include <common_test_utils/test_common.hpp>

#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace ov;

namespace {

struct StreamsCalculationTestCase {
    int input_streams;
    bool input_streams_chaged;
    int input_threads;
    int input_infer_requests;
    int model_prefer_threads;
    std::string input_perf_hint;
    ov::intel_cpu::Config::LatencyThreadingMode latencyThreadingMode;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> stream_info_table;
};

class StreamsCalculationTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<StreamsCalculationTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_stream_info_table =
            ov::intel_cpu::get_streams_info_table(test_data.input_streams,
                                                  test_data.input_streams_chaged,
                                                  test_data.input_threads,
                                                  test_data.input_infer_requests,
                                                  test_data.model_prefer_threads,
                                                  test_data.input_perf_hint,
                                                  test_data.latencyThreadingMode,
                                                  test_data.proc_type_table);

        ASSERT_EQ(test_data.stream_info_table, test_stream_info_table);
    }
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_1 = {
    1,      // param[in]: the number of streams in this simulation.
    false,  // param[in]: Whether the user explicitly sets the number of streams is higher priority in streams
            // calculation logic. If this param is true, the following performance hint and LatencyThreadingMode will be
            // ignored.
    0,      // param[in]: the number of threads in this simulation
    0,      // param[in]: the number of infer requests in this simulation
    0,      // param[in]: the model preferred number of threads in this simulation
    "LATENCY",                                                  // param[in]: the performance hint in this simulation
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,  // param[in]: the LatencyThreadingMode in this
                                                                // simulation
    {{208, 104, 0, 104, -1, -1},
     {104, 52, 0, 52, 0, 0},
     {104, 52, 0, 52, 1, 1}},  // param[in]: the proc_type_table in this simulation
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {-1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},  // param[expected out]: the expected result of streams_info_table in this
                                            // simulation
};

StreamsCalculationTestCase _2sockets_104cores_latency_platform_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0},
     {-1, MAIN_CORE_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_platform_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{1, ALL_PROC, 104, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {0, HYPER_THREADING_PROC, 26, 0, 0},
     {0, HYPER_THREADING_PROC, 26, 1, 0},
     {-1, ALL_PROC, 104, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1},
     {0, HYPER_THREADING_PROC, 26, 2, 1},
     {0, HYPER_THREADING_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_platform_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {26, 26, 0, 0, 0, 0}, {26, 26, 0, 0, 1, 0}, {26, 26, 0, 0, 2, 1}, {26, 26, 0, 0, 3, 1}},
    {{1, ALL_PROC, 52, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {-1, ALL_PROC, 52, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{1, ALL_PROC, 104, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {0, HYPER_THREADING_PROC, 26, 0, 0},
     {0, HYPER_THREADING_PROC, 26, 1, 0},
     {1, ALL_PROC, 104, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1},
     {0, HYPER_THREADING_PROC, 26, 2, 1},
     {0, HYPER_THREADING_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {26, 26, 0, 0, 0, 0}, {26, 26, 0, 0, 1, 0}, {26, 26, 0, 0, 2, 1}, {26, 26, 0, 0, 3, 1}},
    {{1, ALL_PROC, 52, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {1, ALL_PROC, 52, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{60, 60, 0, 0, -1, -1}, {10, 10, 0, 0, 0, 0}, {10, 10, 0, 0, 1, 0}, {20, 20, 0, 0, 2, 1}, {20, 20, 0, 0, 3, 1}},
    {{1, ALL_PROC, 40, -1, 1}, {0, MAIN_CORE_PROC, 20, 2, 1}, {0, MAIN_CORE_PROC, 20, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_6 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{60, 60, 0, 0, -1, -1}, {10, 10, 0, 0, 0, 0}, {20, 20, 0, 0, 1, 1}, {10, 10, 0, 0, 2, 0}, {20, 20, 0, 0, 3, 1}},
    {{1, ALL_PROC, 40, -1, 1}, {0, MAIN_CORE_PROC, 20, 1, 1}, {0, MAIN_CORE_PROC, 20, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_socket_7 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_SOCKET,
    {{104, 104, 0, 0, -1, -1}, {26, 26, 0, 0, 0, 0}, {26, 26, 0, 0, 1, 0}, {26, 26, 0, 0, 2, 1}, {26, 26, 0, 0, 3, 1}},
    {{1, ALL_PROC, 52, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {-1, ALL_PROC, 52, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_node_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_node_2 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_node_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{1, ALL_PROC, 52, 0, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, HYPER_THREADING_PROC, 26, 0, 0},
     {1, ALL_PROC, 52, 1, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {0, HYPER_THREADING_PROC, 26, 1, 0},
     {1, ALL_PROC, 52, 2, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, HYPER_THREADING_PROC, 26, 2, 1},
     {1, ALL_PROC, 52, 3, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1},
     {0, HYPER_THREADING_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_node_4 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {26, 26, 0, 0, 0, 0}, {26, 26, 0, 0, 1, 0}, {26, 26, 0, 0, 2, 1}, {26, 26, 0, 0, 3, 1}},
    {{1, MAIN_CORE_PROC, 26, 0, 0},
     {1, MAIN_CORE_PROC, 26, 1, 0},
     {1, MAIN_CORE_PROC, 26, 2, 1},
     {1, MAIN_CORE_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_node_5 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_NUMA_NODE,
    {{104, 104, 0, 0, -1, -1}, {26, 26, 0, 0, 0, 0}, {26, 26, 0, 0, 1, 0}, {26, 26, 0, 0, 2, 1}, {26, 26, 0, 0, 3, 1}},
    {{1, ALL_PROC, 52, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {-1, ALL_PROC, 52, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_1 = {
    1,
    false,
    20,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_2 = {
    1,
    false,
    20,
    5,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_3 = {
    1,
    false,
    208,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {-1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_4 = {
    1,
    true,
    20,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_5 = {
    1,
    true,
    20,
    5,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_latency_6 = {
    1,
    true,
    208,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {-1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{13, MAIN_CORE_PROC, 4, 0, 0},
     {13, MAIN_CORE_PROC, 4, 1, 1},
     {13, HYPER_THREADING_PROC, 4, 0, 0},
     {13, HYPER_THREADING_PROC, 4, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_3 = {
    1,
    false,
    20,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{5, MAIN_CORE_PROC, 4, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_4 = {
    2,
    true,
    20,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 10, 0, 0}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{52, MAIN_CORE_PROC, 1, 0, 0},
     {52, MAIN_CORE_PROC, 1, 1, 1},
     {52, HYPER_THREADING_PROC, 1, 0, 0},
     {52, HYPER_THREADING_PROC, 1, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{26, MAIN_CORE_PROC, 2, 0, 0},
     {26, MAIN_CORE_PROC, 2, 1, 1},
     {26, HYPER_THREADING_PROC, 2, 0, 0},
     {26, HYPER_THREADING_PROC, 2, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_7 = {
    1,
    false,
    0,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{6, MAIN_CORE_PROC, 8, 0, 0},
     {6, MAIN_CORE_PROC, 8, 1, 1},
     {6, HYPER_THREADING_PROC, 8, 0, 0},
     {6, HYPER_THREADING_PROC, 8, 1, 1},
     {1, ALL_PROC, 8, -1, -1},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, MAIN_CORE_PROC, 4, 1, 1},
     {1, ALL_PROC, 8, -1, -1},
     {0, HYPER_THREADING_PROC, 4, 0, 0},
     {0, HYPER_THREADING_PROC, 4, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_7_1 = {
    26,
    true,
    0,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{6, MAIN_CORE_PROC, 8, 0, 0},
     {6, MAIN_CORE_PROC, 8, 1, 1},
     {6, HYPER_THREADING_PROC, 8, 0, 0},
     {6, HYPER_THREADING_PROC, 8, 1, 1},
     {1, ALL_PROC, 8, -1, -1},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, MAIN_CORE_PROC, 4, 1, 1},
     {1, ALL_PROC, 8, -1, -1},
     {0, HYPER_THREADING_PROC, 4, 0, 0},
     {0, HYPER_THREADING_PROC, 4, 1, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_7_2 = {
    1,
    false,
    0,
    0,
    4,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1},
     {52, 26, 0, 26, 0, 0},
     {52, 26, 0, 26, 1, 0},
     {52, 26, 0, 26, 2, 1},
     {52, 26, 0, 26, 3, 1}},
    {{6, MAIN_CORE_PROC, 4, 0, 0},       {6, MAIN_CORE_PROC, 4, 1, 0},       {6, MAIN_CORE_PROC, 4, 2, 1},
     {6, MAIN_CORE_PROC, 4, 3, 1},       {6, HYPER_THREADING_PROC, 4, 0, 0}, {6, HYPER_THREADING_PROC, 4, 1, 0},
     {6, HYPER_THREADING_PROC, 4, 2, 1}, {6, HYPER_THREADING_PROC, 4, 3, 1}, {1, ALL_PROC, 4, -1, 0},
     {0, MAIN_CORE_PROC, 2, 0, 0},       {0, MAIN_CORE_PROC, 2, 1, 0},       {1, ALL_PROC, 4, -1, 1},
     {0, MAIN_CORE_PROC, 2, 2, 1},       {0, MAIN_CORE_PROC, 2, 3, 1},       {1, ALL_PROC, 4, -1, 0},
     {0, HYPER_THREADING_PROC, 2, 0, 0}, {0, HYPER_THREADING_PROC, 2, 1, 0}, {1, ALL_PROC, 4, -1, 1},
     {0, HYPER_THREADING_PROC, 2, 2, 1}, {0, HYPER_THREADING_PROC, 2, 3, 1}},
};
StreamsCalculationTestCase _2sockets_104cores_tput_8 = {
    1,
    false,
    40,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{5, MAIN_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_9 = {
    5,
    true,
    20,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{2, MAIN_CORE_PROC, 10, 0, 0}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_10 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_11 = {
    2,
    true,
    0,
    5,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_12 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{208, 104, 0, 104, -1, -1}, {104, 52, 0, 52, 0, 0}, {104, 52, 0, 52, 1, 1}},
    {{1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_13 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{13, MAIN_CORE_PROC, 4, 0, 0}, {13, MAIN_CORE_PROC, 4, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_14 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_15 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{52, MAIN_CORE_PROC, 1, 0, 0}, {52, MAIN_CORE_PROC, 1, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_16 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{26, MAIN_CORE_PROC, 2, 0, 0}, {26, MAIN_CORE_PROC, 2, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_17 = {
    1,
    false,
    0,
    0,
    8,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{6, MAIN_CORE_PROC, 8, 0, 0},
     {6, MAIN_CORE_PROC, 8, 1, 1},
     {1, ALL_PROC, 8, -1, -1},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, MAIN_CORE_PROC, 4, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_18 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_19 = {
    2,
    true,
    0,
    5,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_20 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{104, 104, 0, 0, -1, -1}, {52, 52, 0, 0, 0, 0}, {52, 52, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 52, 0, 0}, {1, MAIN_CORE_PROC, 52, 1, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0},
     {-1, MAIN_CORE_PROC, 24, 1, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{6, MAIN_CORE_PROC, 4, 0, 0}, {6, MAIN_CORE_PROC, 4, 1, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_2 = {
    100,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{24, MAIN_CORE_PROC, 1, 0, 0}, {24, MAIN_CORE_PROC, 1, 1, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_3 = {
    1,
    false,
    100,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{6, MAIN_CORE_PROC, 4, 0, 0}, {6, MAIN_CORE_PROC, 4, 1, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_4 = {
    2,
    true,
    20,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{2, MAIN_CORE_PROC, 10, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_2 = {
    1,
    false,
    10,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 10, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_3 = {
    1,
    false,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_4 = {
    1,
    false,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_5 = {
    1,
    false,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_6 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_7 = {
    1,
    false,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_8 = {
    1,
    false,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_9 = {
    1,
    false,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_10 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_11 = {
    1,
    true,
    10,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 10, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_12 = {
    1,
    true,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_13 = {
    1,
    true,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_14 = {
    1,
    true,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_15 = {
    1,
    true,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 20, 0, 0},
     {0, MAIN_CORE_PROC, 6, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_16 = {
    1,
    true,
    0,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_17 = {
    1,
    true,
    0,
    0,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_18 = {
    1,
    true,
    0,
    2,
    14,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{14, 6, 8, 0, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, EFFICIENT_CORE_PROC, 3, 0, 0}, {2, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}, {1, EFFICIENT_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}, {2, EFFICIENT_CORE_PROC, 4, 0, 0}, {1, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_4 = {
    1,
    false,
    12,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, EFFICIENT_CORE_PROC, 3, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {6, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{3, MAIN_CORE_PROC, 2, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {3, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_7 = {
    100,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {8, EFFICIENT_CORE_PROC, 1, 0, 0}, {6, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_8 = {
    1,
    false,
    100,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, EFFICIENT_CORE_PROC, 3, 0, 0}, {2, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_9 = {
    4,
    true,
    0,
    8,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}, {2, EFFICIENT_CORE_PROC, 4, 0, 0}, {1, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_10 = {
    6,
    true,
    0,
    4,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}, {2, EFFICIENT_CORE_PROC, 4, 0, 0}, {1, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_11 = {
    1,
    false,
    0,
    2,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}, {1, EFFICIENT_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_12 = {
    1,
    false,
    0,
    2,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}, {1, EFFICIENT_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_13 = {
    1,
    false,
    1,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_14 = {
    1,
    false,
    9,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {1, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_15 = {
    1,
    false,
    12,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {3, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_16 = {
    1,
    false,
    15,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {1, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_17 = {
    1,
    true,
    14,
    0,
    6,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0},
     {0, MAIN_CORE_PROC, 2, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_2 = {
    1,
    false,
    8,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 8, 0, 0}, {0, MAIN_CORE_PROC, 2, 0, 0}, {0, EFFICIENT_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_3 = {
    1,
    false,
    0,
    0,
    2,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 4, 0, 0}, {0, MAIN_CORE_PROC, 2, 0, 0}, {0, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_4 = {
    1,
    false,
    0,
    0,
    10,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0},
     {0, MAIN_CORE_PROC, 2, 0, 0},
     {0, EFFICIENT_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, ALL_PROC, 10, 0, 0}, {0, MAIN_CORE_PROC, 2, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_6 = {
    1,
    false,
    0,
    0,
    2,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_7 = {
    1,
    false,
    0,
    0,
    10,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{10, 2, 8, 0, 0, 0}},
    {{1, ALL_PROC, 10, 0, 0}, {0, MAIN_CORE_PROC, 2, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {1, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}, {1, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}, {3, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_4 = {
    1,
    false,
    6,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_5 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{2, MAIN_CORE_PROC, 1, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {2, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_6 = {
    1,
    false,
    0,
    0,
    2,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 2, 8, 2, 0, 0}},
    {{1, MAIN_CORE_PROC, 2, 0, 0}, {4, EFFICIENT_CORE_PROC, 2, 0, 0}, {1, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, EFFICIENT_CORE_PROC, 4, 0, 0},
     {0, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_2 = {
    1,
    false,
    100,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, EFFICIENT_CORE_PROC, 4, 0, 0},
     {0, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_3 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 8, 0, 0}, {0, MAIN_CORE_PROC, 4, 0, 0}, {0, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_4 = {
    1,
    false,
    0,
    0,
    8,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0},
     {0, MAIN_CORE_PROC, 4, 0, 0},
     {0, EFFICIENT_CORE_PROC, 4, 0, 0},
     {0, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_5 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, ALL_PROC, 8, 0, 0}, {0, MAIN_CORE_PROC, 4, 0, 0}, {0, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_6 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_7 = {
    1,
    false,
    0,
    0,
    8,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{1, ALL_PROC, 8, 0, 0}, {0, MAIN_CORE_PROC, 4, 0, 0}, {0, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}, {1, EFFICIENT_CORE_PROC, 4, 0, 0}, {1, HYPER_THREADING_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{1, MAIN_CORE_PROC, 4, 0, 0}, {1, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_3 = {
    4,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_4 = {
    6,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}, {2, HYPER_THREADING_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_5 = {
    1,
    false,
    6,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2, 0, 0}, {1, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_6 = {
    1,
    false,
    8,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{2, MAIN_CORE_PROC, 2, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_7 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 4, 4, 4, 0, 0}},
    {{4, MAIN_CORE_PROC, 1, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}, {4, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_8 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{8, 4, 4, 0, 0, 0}},
    {{2, MAIN_CORE_PROC, 2, 0, 0}, {2, EFFICIENT_CORE_PROC, 2, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_2 = {
    1,
    false,
    100,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, ALL_PROC, 12, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_3 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{6, 6, 0, 0, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_1 = {
    1,
    false,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}, {2, HYPER_THREADING_PROC, 3, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_2 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{1, MAIN_CORE_PROC, 6, 0, 0}, {1, HYPER_THREADING_PROC, 6, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_3 = {
    1,
    false,
    8,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{2, MAIN_CORE_PROC, 3, 0, 0}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_4 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{12, 6, 0, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {6, HYPER_THREADING_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_1 = {
    1,
    false,
    0,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 16, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_2 = {
    1,
    false,
    4,
    0,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_3 = {
    1,
    false,
    0,
    4,
    0,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 16, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_latency_4 = {
    1,
    false,
    0,
    0,
    4,
    "LATENCY",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{1, EFFICIENT_CORE_PROC, 16, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_1 = {
    1,
    false,
    0,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{16, EFFICIENT_CORE_PROC, 1, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_2 = {
    1,
    false,
    0,
    0,
    4,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{4, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_3 = {
    2,
    true,
    0,
    0,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{2, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_4 = {
    8,
    true,
    0,
    4,
    0,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{4, EFFICIENT_CORE_PROC, 4, 0, 0}},
};

StreamsCalculationTestCase _1sockets_ecores_tput_5 = {
    2,
    true,
    0,
    0,
    4,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{16, 0, 16, 0, 0, 0}},
    {{2, EFFICIENT_CORE_PROC, 8, 0, 0}},
};

StreamsCalculationTestCase _1sockets_mock_tput_1 = {
    1,
    false,
    15,
    0,
    1,
    "THROUGHPUT",
    ov::intel_cpu::Config::LatencyThreadingMode::PER_PLATFORM,
    {{20, 6, 7, 6, 0, 0}},
    {{6, MAIN_CORE_PROC, 1, 0, 0}, {3, EFFICIENT_CORE_PROC, 2, 0, 0}, {3, HYPER_THREADING_PROC, 1, 0, 0}},
};

TEST_P(StreamsCalculationTests, StreamsCalculation) {}

INSTANTIATE_TEST_SUITE_P(StreamsInfoTable,
                         StreamsCalculationTests,
                         testing::Values(_2sockets_104cores_latency_platform_1,
                                         _2sockets_104cores_latency_platform_2,
                                         _2sockets_104cores_latency_platform_3,
                                         _2sockets_104cores_latency_platform_4,
                                         _2sockets_104cores_latency_socket_1,
                                         _2sockets_104cores_latency_socket_2,
                                         _2sockets_104cores_latency_socket_3,
                                         _2sockets_104cores_latency_socket_4,
                                         _2sockets_104cores_latency_socket_5,
                                         _2sockets_104cores_latency_socket_6,
                                         _2sockets_104cores_latency_socket_7,
                                         _2sockets_104cores_latency_node_1,
                                         _2sockets_104cores_latency_node_2,
                                         _2sockets_104cores_latency_node_3,
                                         _2sockets_104cores_latency_node_4,
                                         _2sockets_104cores_latency_node_5,
                                         _2sockets_104cores_latency_1,
                                         _2sockets_104cores_latency_2,
                                         _2sockets_104cores_latency_3,
                                         _2sockets_104cores_latency_4,
                                         _2sockets_104cores_latency_5,
                                         _2sockets_104cores_latency_6,
                                         _2sockets_104cores_tput_1,
                                         _2sockets_104cores_tput_2,
                                         _2sockets_104cores_tput_3,
                                         _2sockets_104cores_tput_4,
                                         _2sockets_104cores_tput_5,
                                         _2sockets_104cores_tput_6,
                                         _2sockets_104cores_tput_7,
                                         _2sockets_104cores_tput_7_1,
                                         _2sockets_104cores_tput_7_2,
                                         _2sockets_104cores_tput_8,
                                         _2sockets_104cores_tput_9,
                                         _2sockets_104cores_tput_10,
                                         _2sockets_104cores_tput_11,
                                         _2sockets_104cores_tput_12,
                                         _2sockets_104cores_tput_13,
                                         _2sockets_104cores_tput_14,
                                         _2sockets_104cores_tput_15,
                                         _2sockets_104cores_tput_16,
                                         _2sockets_104cores_tput_17,
                                         _2sockets_104cores_tput_18,
                                         _2sockets_104cores_tput_19,
                                         _2sockets_104cores_tput_20,
                                         _2sockets_48cores_latency_1,
                                         _2sockets_48cores_tput_1,
                                         _2sockets_48cores_tput_2,
                                         _2sockets_48cores_tput_3,
                                         _2sockets_48cores_tput_4,
                                         _1sockets_14cores_latency_1,
                                         _1sockets_14cores_latency_2,
                                         _1sockets_14cores_latency_3,
                                         _1sockets_14cores_latency_4,
                                         _1sockets_14cores_latency_5,
                                         _1sockets_14cores_latency_6,
                                         _1sockets_14cores_latency_7,
                                         _1sockets_14cores_latency_8,
                                         _1sockets_14cores_latency_9,
                                         _1sockets_14cores_latency_10,
                                         _1sockets_14cores_latency_11,
                                         _1sockets_14cores_latency_12,
                                         _1sockets_14cores_latency_13,
                                         _1sockets_14cores_latency_14,
                                         _1sockets_14cores_latency_15,
                                         _1sockets_14cores_latency_16,
                                         _1sockets_14cores_latency_17,
                                         _1sockets_14cores_latency_18,
                                         _1sockets_14cores_tput_1,
                                         _1sockets_14cores_tput_2,
                                         _1sockets_14cores_tput_3,
                                         _1sockets_14cores_tput_4,
                                         _1sockets_14cores_tput_5,
                                         _1sockets_14cores_tput_6,
                                         _1sockets_14cores_tput_7,
                                         _1sockets_14cores_tput_8,
                                         _1sockets_14cores_tput_9,
                                         _1sockets_14cores_tput_10,
                                         _1sockets_14cores_tput_11,
                                         _1sockets_14cores_tput_12,
                                         _1sockets_14cores_tput_13,
                                         _1sockets_14cores_tput_14,
                                         _1sockets_14cores_tput_15,
                                         _1sockets_14cores_tput_16,
                                         _1sockets_14cores_tput_17,
                                         _1sockets_10cores_latency_1,
                                         _1sockets_10cores_latency_2,
                                         _1sockets_10cores_latency_3,
                                         _1sockets_10cores_latency_4,
                                         _1sockets_10cores_latency_5,
                                         _1sockets_10cores_latency_6,
                                         _1sockets_10cores_latency_7,
                                         _1sockets_10cores_tput_1,
                                         _1sockets_10cores_tput_2,
                                         _1sockets_10cores_tput_3,
                                         _1sockets_10cores_tput_4,
                                         _1sockets_10cores_tput_5,
                                         _1sockets_10cores_tput_6,
                                         _1sockets_8cores_latency_1,
                                         _1sockets_8cores_latency_2,
                                         _1sockets_8cores_latency_3,
                                         _1sockets_8cores_latency_4,
                                         _1sockets_8cores_latency_5,
                                         _1sockets_8cores_latency_6,
                                         _1sockets_8cores_latency_7,
                                         _1sockets_8cores_tput_1,
                                         _1sockets_8cores_tput_2,
                                         _1sockets_8cores_tput_3,
                                         _1sockets_8cores_tput_4,
                                         _1sockets_8cores_tput_5,
                                         _1sockets_8cores_tput_6,
                                         _1sockets_8cores_tput_7,
                                         _1sockets_8cores_tput_8,
                                         _1sockets_6cores_latency_1,
                                         _1sockets_6cores_latency_2,
                                         _1sockets_6cores_latency_3,
                                         _1sockets_6cores_tput_1,
                                         _1sockets_6cores_tput_2,
                                         _1sockets_6cores_tput_3,
                                         _1sockets_6cores_tput_4,
                                         _1sockets_ecores_latency_1,
                                         _1sockets_ecores_latency_2,
                                         _1sockets_ecores_latency_3,
                                         _1sockets_ecores_latency_4,
                                         _1sockets_ecores_tput_1,
                                         _1sockets_ecores_tput_2,
                                         _1sockets_ecores_tput_3,
                                         _1sockets_ecores_tput_4,
                                         _1sockets_ecores_tput_5,
                                         _1sockets_mock_tput_1));

}  // namespace