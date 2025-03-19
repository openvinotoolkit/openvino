// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_executor_internal.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;
using namespace threading;

namespace {

#if defined(__linux__) || defined(_WIN32)

#    define NUMA_ALL -1

struct LinuxCpuStreamTypeCase {
    bool _cpu_pinning;
    int _numa_nodes;
    std::vector<std::vector<int>> _cpu_mapping_table;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _streams_info_table;
    std::vector<StreamCreateType> _stream_type;
    std::vector<int> _concurrency;
    std::vector<int> _core_type;
    std::vector<int> _numa_node_id;
    std::vector<int> _max_threads_per_core;
};

class LinuxCpuStreamTypeTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxCpuStreamTypeCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> stream_processor_ids;
        std::vector<StreamCreateType> test_stream_types;
        std::vector<int> test_concurrencys;
        std::vector<int> test_core_types;
        std::vector<int> test_numa_node_ids;
        std::vector<int> test_max_threads_per_cores;
        int streams = 0;

        for (size_t i = 0; i < test_data._streams_info_table.size(); i++) {
            streams += test_data._streams_info_table[i][NUMBER_OF_STREAMS];
        }

        ov::threading::reserve_cpu_by_streams_info(test_data._streams_info_table,
                                                   test_data._numa_nodes,
                                                   test_data._cpu_mapping_table,
                                                   test_data._proc_type_table,
                                                   stream_processor_ids,
                                                   NOT_USED);

        for (auto i = 0; i < streams; i++) {
            StreamCreateType test_stream_type;
            int test_concurrency;
            int test_core_type;
            int test_numa_node_id;
            int test_socket_id = 0;
            int test_max_threads_per_core;
            get_cur_stream_info(i,
                                test_data._cpu_pinning,
                                test_data._proc_type_table,
                                test_data._streams_info_table,
                                test_stream_type,
                                test_concurrency,
                                test_core_type,
                                test_numa_node_id,
                                test_socket_id,
                                test_max_threads_per_core);
            test_stream_types.push_back(test_stream_type);
            test_concurrencys.push_back(test_concurrency);
            test_core_types.push_back(test_core_type);
            test_numa_node_ids.push_back(test_numa_node_id);
            test_max_threads_per_cores.push_back(test_max_threads_per_core);
        }

        ASSERT_EQ(test_data._stream_type, test_stream_types);
        ASSERT_EQ(test_data._concurrency, test_concurrencys);
        ASSERT_EQ(test_data._core_type, test_core_types);
        ASSERT_EQ(test_data._numa_node_id, test_numa_node_ids);
        ASSERT_EQ(test_data._max_threads_per_core, test_max_threads_per_cores);
    }
};

LinuxCpuStreamTypeCase _2sockets_72cores_nobinding_36streams = {
    false,  // param[in]: cpu_pinning
    2,      // param[in]: number of numa nodes
    // param[in]: cpu_mapping_table, {PROCESSOR_ID, SOCKET_ID, CORE_ID, CORE_TYPE, GROUP_ID, Used}
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {3, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {5, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},    {7, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 8, HYPER_THREADING_PROC, 8, -1},    {9, 0, 0, 9, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 10, HYPER_THREADING_PROC, 10, -1}, {11, 0, 0, 11, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 12, HYPER_THREADING_PROC, 12, -1}, {13, 0, 0, 13, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 14, HYPER_THREADING_PROC, 14, -1}, {15, 0, 0, 15, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 16, HYPER_THREADING_PROC, 16, -1}, {17, 0, 0, 17, HYPER_THREADING_PROC, 17, -1},
        {18, 1, 1, 18, HYPER_THREADING_PROC, 18, -1}, {19, 1, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 1, 1, 20, HYPER_THREADING_PROC, 20, -1}, {21, 1, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {22, 1, 1, 22, HYPER_THREADING_PROC, 22, -1}, {23, 1, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 1, 1, 24, HYPER_THREADING_PROC, 24, -1}, {25, 1, 1, 25, HYPER_THREADING_PROC, 25, -1},
        {26, 1, 1, 26, HYPER_THREADING_PROC, 26, -1}, {27, 1, 1, 27, HYPER_THREADING_PROC, 27, -1},
        {28, 1, 1, 28, HYPER_THREADING_PROC, 28, -1}, {29, 1, 1, 29, HYPER_THREADING_PROC, 29, -1},
        {30, 1, 1, 30, HYPER_THREADING_PROC, 30, -1}, {31, 1, 1, 31, HYPER_THREADING_PROC, 31, -1},
        {32, 1, 1, 32, HYPER_THREADING_PROC, 32, -1}, {33, 1, 1, 33, HYPER_THREADING_PROC, 33, -1},
        {34, 1, 1, 34, HYPER_THREADING_PROC, 34, -1}, {35, 1, 1, 35, HYPER_THREADING_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
        {40, 0, 0, 40, MAIN_CORE_PROC, 40, -1},       {41, 0, 0, 41, MAIN_CORE_PROC, 41, -1},
        {42, 0, 0, 42, MAIN_CORE_PROC, 42, -1},       {43, 0, 0, 43, MAIN_CORE_PROC, 43, -1},
        {44, 0, 0, 44, MAIN_CORE_PROC, 44, -1},       {45, 0, 0, 45, MAIN_CORE_PROC, 45, -1},
        {46, 0, 0, 46, MAIN_CORE_PROC, 46, -1},       {47, 0, 0, 47, MAIN_CORE_PROC, 47, -1},
        {48, 0, 0, 48, MAIN_CORE_PROC, 48, -1},       {49, 0, 0, 49, MAIN_CORE_PROC, 49, -1},
        {50, 0, 0, 50, MAIN_CORE_PROC, 50, -1},       {51, 0, 0, 51, MAIN_CORE_PROC, 51, -1},
        {52, 0, 0, 52, MAIN_CORE_PROC, 52, -1},       {53, 0, 0, 53, MAIN_CORE_PROC, 53, -1},
        {54, 1, 1, 54, MAIN_CORE_PROC, 54, -1},       {55, 1, 1, 55, MAIN_CORE_PROC, 55, -1},
        {56, 1, 1, 56, MAIN_CORE_PROC, 56, -1},       {57, 1, 1, 57, MAIN_CORE_PROC, 57, -1},
        {58, 1, 1, 58, MAIN_CORE_PROC, 58, -1},       {59, 1, 1, 59, MAIN_CORE_PROC, 59, -1},
        {60, 1, 1, 60, MAIN_CORE_PROC, 60, -1},       {61, 1, 1, 61, MAIN_CORE_PROC, 61, -1},
        {62, 1, 1, 62, MAIN_CORE_PROC, 62, -1},       {63, 1, 1, 63, MAIN_CORE_PROC, 63, -1},
        {64, 1, 1, 64, MAIN_CORE_PROC, 64, -1},       {65, 1, 1, 65, MAIN_CORE_PROC, 65, -1},
        {66, 1, 1, 66, MAIN_CORE_PROC, 66, -1},       {67, 1, 1, 67, MAIN_CORE_PROC, 67, -1},
        {68, 1, 1, 68, MAIN_CORE_PROC, 68, -1},       {69, 1, 1, 69, MAIN_CORE_PROC, 69, -1},
        {70, 1, 1, 70, MAIN_CORE_PROC, 70, -1},       {71, 1, 1, 71, MAIN_CORE_PROC, 71, -1},
    },
    // param[in]: proc_type_table,
    {{72, 36, 0, 36, -1, -1}, {36, 18, 0, 18, 0, 0}, {36, 18, 0, 18, 1, 1}},
    // param[in]: streams_info_table, {NUMBER_OF_STREAMS, PROC_TYPE, THREADS_PER_STREAM}
    {{18, MAIN_CORE_PROC, 1, 0, 0}, {18, MAIN_CORE_PROC, 1, 1, 1}},
    // param[out]: stream_type per stream used in new task_arena
    {
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID, STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
    },
    // param[out]: concurrency per stream used in new task_arena
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // param[out]: core_type per stream used in new task_arena
    {
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
        MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC, MAIN_CORE_PROC,
    },
    // param[out]: numa_node_id per stream used in new task_arena
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    // param[out]: max_threads_per_core per stream used in new task_arena
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
};
LinuxCpuStreamTypeCase _2sockets_72cores_nobinding_9streams = {
    false,
    2,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {3, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {5, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},    {7, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 8, HYPER_THREADING_PROC, 8, -1},    {9, 0, 0, 9, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 10, HYPER_THREADING_PROC, 10, -1}, {11, 0, 0, 11, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 12, HYPER_THREADING_PROC, 12, -1}, {13, 0, 0, 13, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 14, HYPER_THREADING_PROC, 14, -1}, {15, 0, 0, 15, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 16, HYPER_THREADING_PROC, 16, -1}, {17, 0, 0, 17, HYPER_THREADING_PROC, 17, -1},
        {18, 1, 1, 18, HYPER_THREADING_PROC, 18, -1}, {19, 1, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 1, 1, 20, HYPER_THREADING_PROC, 20, -1}, {21, 1, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {22, 1, 1, 22, HYPER_THREADING_PROC, 22, -1}, {23, 1, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 1, 1, 24, HYPER_THREADING_PROC, 24, -1}, {25, 1, 1, 25, HYPER_THREADING_PROC, 25, -1},
        {26, 1, 1, 26, HYPER_THREADING_PROC, 26, -1}, {27, 1, 1, 27, HYPER_THREADING_PROC, 27, -1},
        {28, 1, 1, 28, HYPER_THREADING_PROC, 28, -1}, {29, 1, 1, 29, HYPER_THREADING_PROC, 29, -1},
        {30, 1, 1, 30, HYPER_THREADING_PROC, 30, -1}, {31, 1, 1, 31, HYPER_THREADING_PROC, 31, -1},
        {32, 1, 1, 32, HYPER_THREADING_PROC, 32, -1}, {33, 1, 1, 33, HYPER_THREADING_PROC, 33, -1},
        {34, 1, 1, 34, HYPER_THREADING_PROC, 34, -1}, {35, 1, 1, 35, HYPER_THREADING_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
        {40, 0, 0, 40, MAIN_CORE_PROC, 40, -1},       {41, 0, 0, 41, MAIN_CORE_PROC, 41, -1},
        {42, 0, 0, 42, MAIN_CORE_PROC, 42, -1},       {43, 0, 0, 43, MAIN_CORE_PROC, 43, -1},
        {44, 0, 0, 44, MAIN_CORE_PROC, 44, -1},       {45, 0, 0, 45, MAIN_CORE_PROC, 45, -1},
        {46, 0, 0, 46, MAIN_CORE_PROC, 46, -1},       {47, 0, 0, 47, MAIN_CORE_PROC, 47, -1},
        {48, 0, 0, 48, MAIN_CORE_PROC, 48, -1},       {49, 0, 0, 49, MAIN_CORE_PROC, 49, -1},
        {50, 0, 0, 50, MAIN_CORE_PROC, 50, -1},       {51, 0, 0, 51, MAIN_CORE_PROC, 51, -1},
        {52, 0, 0, 52, MAIN_CORE_PROC, 52, -1},       {53, 0, 0, 53, MAIN_CORE_PROC, 53, -1},
        {54, 1, 1, 54, MAIN_CORE_PROC, 54, -1},       {55, 1, 1, 55, MAIN_CORE_PROC, 55, -1},
        {56, 1, 1, 56, MAIN_CORE_PROC, 56, -1},       {57, 1, 1, 57, MAIN_CORE_PROC, 57, -1},
        {58, 1, 1, 58, MAIN_CORE_PROC, 58, -1},       {59, 1, 1, 59, MAIN_CORE_PROC, 59, -1},
        {60, 1, 1, 60, MAIN_CORE_PROC, 60, -1},       {61, 1, 1, 61, MAIN_CORE_PROC, 61, -1},
        {62, 1, 1, 62, MAIN_CORE_PROC, 62, -1},       {63, 1, 1, 63, MAIN_CORE_PROC, 63, -1},
        {64, 1, 1, 64, MAIN_CORE_PROC, 64, -1},       {65, 1, 1, 65, MAIN_CORE_PROC, 65, -1},
        {66, 1, 1, 66, MAIN_CORE_PROC, 66, -1},       {67, 1, 1, 67, MAIN_CORE_PROC, 67, -1},
        {68, 1, 1, 68, MAIN_CORE_PROC, 68, -1},       {69, 1, 1, 69, MAIN_CORE_PROC, 69, -1},
        {70, 1, 1, 70, MAIN_CORE_PROC, 70, -1},       {71, 1, 1, 71, MAIN_CORE_PROC, 71, -1},
    },
    {{72, 36, 0, 36, -1, -1}, {36, 18, 0, 18, 0, 0}, {36, 18, 0, 18, 1, 1}},
    {{4, MAIN_CORE_PROC, 4, 0, 0},
     {4, MAIN_CORE_PROC, 4, 1, 1},
     {1, ALL_PROC, 4, -1, -1},
     {0, MAIN_CORE_PROC, 2, 0, 0},
     {0, MAIN_CORE_PROC, 2, 1, 1}},
    {
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITH_NUMA_ID,
        STREAM_WITHOUT_PARAM,
    },
    {4, 4, 4, 4, 4, 4, 4, 4, 4},
    {
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        ALL_PROC,
    },
    {0, 0, 0, 0, 1, 1, 1, 1, NUMA_ALL},
    {1, 1, 1, 1, 1, 1, 1, 1, 1},
};
LinuxCpuStreamTypeCase _2sockets_72cores_binding_9streams = {
    true,
    2,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {3, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {5, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},    {7, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 8, HYPER_THREADING_PROC, 8, -1},    {9, 0, 0, 9, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 10, HYPER_THREADING_PROC, 10, -1}, {11, 0, 0, 11, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 12, HYPER_THREADING_PROC, 12, -1}, {13, 0, 0, 13, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 14, HYPER_THREADING_PROC, 14, -1}, {15, 0, 0, 15, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 16, HYPER_THREADING_PROC, 16, -1}, {17, 0, 0, 17, HYPER_THREADING_PROC, 17, -1},
        {18, 1, 1, 18, HYPER_THREADING_PROC, 18, -1}, {19, 1, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 1, 1, 20, HYPER_THREADING_PROC, 20, -1}, {21, 1, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {22, 1, 1, 22, HYPER_THREADING_PROC, 22, -1}, {23, 1, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 1, 1, 24, HYPER_THREADING_PROC, 24, -1}, {25, 1, 1, 25, HYPER_THREADING_PROC, 25, -1},
        {26, 1, 1, 26, HYPER_THREADING_PROC, 26, -1}, {27, 1, 1, 27, HYPER_THREADING_PROC, 27, -1},
        {28, 1, 1, 28, HYPER_THREADING_PROC, 28, -1}, {29, 1, 1, 29, HYPER_THREADING_PROC, 29, -1},
        {30, 1, 1, 30, HYPER_THREADING_PROC, 30, -1}, {31, 1, 1, 31, HYPER_THREADING_PROC, 31, -1},
        {32, 1, 1, 32, HYPER_THREADING_PROC, 32, -1}, {33, 1, 1, 33, HYPER_THREADING_PROC, 33, -1},
        {34, 1, 1, 34, HYPER_THREADING_PROC, 34, -1}, {35, 1, 1, 35, HYPER_THREADING_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
        {40, 0, 0, 40, MAIN_CORE_PROC, 40, -1},       {41, 0, 0, 41, MAIN_CORE_PROC, 41, -1},
        {42, 0, 0, 42, MAIN_CORE_PROC, 42, -1},       {43, 0, 0, 43, MAIN_CORE_PROC, 43, -1},
        {44, 0, 0, 44, MAIN_CORE_PROC, 44, -1},       {45, 0, 0, 45, MAIN_CORE_PROC, 45, -1},
        {46, 0, 0, 46, MAIN_CORE_PROC, 46, -1},       {47, 0, 0, 47, MAIN_CORE_PROC, 47, -1},
        {48, 0, 0, 48, MAIN_CORE_PROC, 48, -1},       {49, 0, 0, 49, MAIN_CORE_PROC, 49, -1},
        {50, 0, 0, 50, MAIN_CORE_PROC, 50, -1},       {51, 0, 0, 51, MAIN_CORE_PROC, 51, -1},
        {52, 0, 0, 52, MAIN_CORE_PROC, 52, -1},       {53, 0, 0, 53, MAIN_CORE_PROC, 53, -1},
        {54, 1, 1, 54, MAIN_CORE_PROC, 54, -1},       {55, 1, 1, 55, MAIN_CORE_PROC, 55, -1},
        {56, 1, 1, 56, MAIN_CORE_PROC, 56, -1},       {57, 1, 1, 57, MAIN_CORE_PROC, 57, -1},
        {58, 1, 1, 58, MAIN_CORE_PROC, 58, -1},       {59, 1, 1, 59, MAIN_CORE_PROC, 59, -1},
        {60, 1, 1, 60, MAIN_CORE_PROC, 60, -1},       {61, 1, 1, 61, MAIN_CORE_PROC, 61, -1},
        {62, 1, 1, 62, MAIN_CORE_PROC, 62, -1},       {63, 1, 1, 63, MAIN_CORE_PROC, 63, -1},
        {64, 1, 1, 64, MAIN_CORE_PROC, 64, -1},       {65, 1, 1, 65, MAIN_CORE_PROC, 65, -1},
        {66, 1, 1, 66, MAIN_CORE_PROC, 66, -1},       {67, 1, 1, 67, MAIN_CORE_PROC, 67, -1},
        {68, 1, 1, 68, MAIN_CORE_PROC, 68, -1},       {69, 1, 1, 69, MAIN_CORE_PROC, 69, -1},
        {70, 1, 1, 70, MAIN_CORE_PROC, 70, -1},       {71, 1, 1, 71, MAIN_CORE_PROC, 71, -1},
    },
    {{72, 36, 0, 36, -1, -1}, {36, 18, 0, 18, 0, 0}, {36, 18, 0, 18, 1, 1}},
    {{4, MAIN_CORE_PROC, 4, 0, 0},
     {4, MAIN_CORE_PROC, 4, 1, 1},
     {1, ALL_PROC, 4, -1, -1},
     {0, MAIN_CORE_PROC, 2, 0, 0},
     {0, MAIN_CORE_PROC, 2, 1, 1}},
    {
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
    },
    {4, 4, 4, 4, 4, 4, 4, 4, 4},
    {
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        ALL_PROC,
    },
    {0, 0, 0, 0, 1, 1, 1, 1, NUMA_ALL},
    {1, 1, 1, 1, 1, 1, 1, 1, 1},
};
LinuxCpuStreamTypeCase _1sockets_4cores_nobinding = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {3, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {5, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {6, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
    },
    {{8, 4, 0, 4, 0, 0}},
    {{1, ALL_PROC, 8, 0, 0}, {0, MAIN_CORE_PROC, 4, 0, 0}, {0, HYPER_THREADING_PROC, 4, 0, 0}},
    {STREAM_WITHOUT_PARAM},
    {8},
    {ALL_PROC},
    {0},
    {2},
};
LinuxCpuStreamTypeCase _1sockets_4cores_binding = {
    true,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {3, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {5, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {6, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
    },
    {{8, 4, 0, 4, 0, 0}},
    {{4, MAIN_CORE_PROC, 1, 0, 0}},
    {
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
    },
    {1, 1, 1, 1},
    {
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
    },
    {0, 0, 0, 0},
    {1, 1, 1, 1},
};

LinuxCpuStreamTypeCase _1sockets_12cores_pcore_nobinding = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{1, MAIN_CORE_PROC, 8, 0, 0}},
    {STREAM_WITH_CORE_TYPE},
    {8},
    {MAIN_CORE_PROC},
    {0},
    {1},
};
LinuxCpuStreamTypeCase _1sockets_12cores_pcore_nobinding_hyper_threading = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{1, ALL_PROC, 16, 0, 0}, {0, MAIN_CORE_PROC, 8, 0, 0}, {0, HYPER_THREADING_PROC, 8, 0, 0}},
    {STREAM_WITH_CORE_TYPE},
    {16},
    {MAIN_CORE_PROC},
    {0},
    {2},
};
LinuxCpuStreamTypeCase _1sockets_12cores_pcore_binding = {
    true,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{2, MAIN_CORE_PROC, 4, 0, 0}},
    {
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
    },
    {4, 4},
    {
        MAIN_CORE_PROC,
        MAIN_CORE_PROC,
    },
    {0, 0},
    {1, 1},
};
LinuxCpuStreamTypeCase _1sockets_12cores_pcore_binding_hyper_threading = {
    true,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{1, MAIN_CORE_PROC, 8, 0, 0}, {1, HYPER_THREADING_PROC, 8, 0, 0}},
    {
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
    },
    {8, 8},
    {
        MAIN_CORE_PROC,
        HYPER_THREADING_PROC,
    },
    {0, 0},
    {1, 2},
};
LinuxCpuStreamTypeCase _1sockets_12cores_ecore_nobinding = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{2, EFFICIENT_CORE_PROC, 2, 0, 0}},
    {
        STREAM_WITH_CORE_TYPE,
        STREAM_WITH_CORE_TYPE,
    },
    {2, 2},
    {
        EFFICIENT_CORE_PROC,
        EFFICIENT_CORE_PROC,
    },
    {0, 0},
    {1, 1},
};
LinuxCpuStreamTypeCase _1sockets_12cores_ecore_binding = {
    true,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1}, {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1}, {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
    },
    {{20, 8, 4, 8, 0, 0}},
    {{4, EFFICIENT_CORE_PROC, 1, 0, 0}},
    {
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
        STREAM_WITH_OBSERVE,
    },
    {1, 1, 1, 1},
    {
        EFFICIENT_CORE_PROC,
        EFFICIENT_CORE_PROC,
        EFFICIENT_CORE_PROC,
        EFFICIENT_CORE_PROC,
    },
    {0, 0, 0, 0},
    {1, 1, 1, 1},
};
LinuxCpuStreamTypeCase _1sockets_24cores_all_proc = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},   {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},   {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},   {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},   {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},   {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},  {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},  {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},  {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},   {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1},  {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 9, -1},  {21, 0, 0, 13, EFFICIENT_CORE_PROC, 9, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 9, -1},  {23, 0, 0, 15, EFFICIENT_CORE_PROC, 9, -1},
        {24, 0, 0, 16, EFFICIENT_CORE_PROC, 10, -1}, {25, 0, 0, 17, EFFICIENT_CORE_PROC, 10, -1},
        {26, 0, 0, 18, EFFICIENT_CORE_PROC, 10, -1}, {27, 0, 0, 19, EFFICIENT_CORE_PROC, 10, -1},
        {28, 0, 0, 20, EFFICIENT_CORE_PROC, 11, -1}, {29, 0, 0, 21, EFFICIENT_CORE_PROC, 11, -1},
        {30, 0, 0, 22, EFFICIENT_CORE_PROC, 11, -1}, {31, 0, 0, 23, EFFICIENT_CORE_PROC, 11, -1},
    },
    {{32, 8, 16, 8, 0, 0}},
    {{1, ALL_PROC, 24, 0, 0}, {0, MAIN_CORE_PROC, 8, 0, 0}, {0, EFFICIENT_CORE_PROC, 16, 0, 0}},
    {STREAM_WITHOUT_PARAM},
    {24},
    {ALL_PROC},
    {0},
    {1},
};
LinuxCpuStreamTypeCase _1sockets_24cores_all_proc_hyper_threading = {
    false,
    1,
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},   {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},   {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},   {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},   {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},   {9, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},  {11, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},  {13, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},  {15, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},   {17, 0, 0, 9, EFFICIENT_CORE_PROC, 8, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 8, -1},  {19, 0, 0, 11, EFFICIENT_CORE_PROC, 8, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 9, -1},  {21, 0, 0, 13, EFFICIENT_CORE_PROC, 9, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 9, -1},  {23, 0, 0, 15, EFFICIENT_CORE_PROC, 9, -1},
        {24, 0, 0, 16, EFFICIENT_CORE_PROC, 10, -1}, {25, 0, 0, 17, EFFICIENT_CORE_PROC, 10, -1},
        {26, 0, 0, 18, EFFICIENT_CORE_PROC, 10, -1}, {27, 0, 0, 19, EFFICIENT_CORE_PROC, 10, -1},
        {28, 0, 0, 20, EFFICIENT_CORE_PROC, 11, -1}, {29, 0, 0, 21, EFFICIENT_CORE_PROC, 11, -1},
        {30, 0, 0, 22, EFFICIENT_CORE_PROC, 11, -1}, {31, 0, 0, 23, EFFICIENT_CORE_PROC, 11, -1},
    },
    {{32, 8, 16, 8, 0, 0}},
    {{1, ALL_PROC, 32, 0, 0},
     {0, MAIN_CORE_PROC, 8, 0, 0},
     {0, HYPER_THREADING_PROC, 8, 0, 0},
     {0, EFFICIENT_CORE_PROC, 16, 0, 0}},
    {STREAM_WITHOUT_PARAM},
    {32},
    {ALL_PROC},
    {0},
    {2},
};

TEST_P(LinuxCpuStreamTypeTests, LinuxCpuStreamType) {}

INSTANTIATE_TEST_SUITE_P(CpuStreamType,
                         LinuxCpuStreamTypeTests,
                         testing::Values(_2sockets_72cores_nobinding_36streams,
                                         _2sockets_72cores_nobinding_9streams,
                                         _2sockets_72cores_binding_9streams,
                                         _1sockets_4cores_nobinding,
                                         _1sockets_4cores_binding,
                                         _1sockets_12cores_pcore_nobinding,
                                         _1sockets_12cores_pcore_nobinding_hyper_threading,
                                         _1sockets_12cores_pcore_binding,
                                         _1sockets_12cores_pcore_binding_hyper_threading,
                                         _1sockets_12cores_ecore_nobinding,
                                         _1sockets_12cores_ecore_binding,
                                         _1sockets_24cores_all_proc,
                                         _1sockets_24cores_all_proc_hyper_threading));
#endif
}  // namespace
