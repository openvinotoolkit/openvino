// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;
using namespace threading;

namespace {

#if defined(__linux__) || defined(_WIN32)

struct ExecutorConfigTestCase {
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
    int _num_streams;
    int _threads_per_stream;
    ov::hint::SchedulingCoreType _core_type;
    bool _cpu_pinning;
    std::vector<std::vector<int>> _streams_info_table_in;
    std::vector<std::vector<int>> _streams_info_table;
    std::vector<std::vector<int>> _stream_processors;
};

class ExecutorConfigTest : public ov::test::TestsCommon,
                           public testing::WithParamInterface<std::tuple<ExecutorConfigTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        CPU& cpu = cpu_info();
        cpu._org_proc_type_table = test_data._proc_type_table;
        cpu._proc_type_table = test_data._proc_type_table;
        cpu._cpu_mapping_table = test_data._cpu_mapping_table;
        cpu._numa_nodes = cpu._proc_type_table.size() > 1 ? static_cast<int>(cpu._proc_type_table.size()) - 1 : 1;
        cpu._sockets = cpu._numa_nodes;

        ov::threading::IStreamsExecutor::Config config{"config test",
                                                       test_data._num_streams,
                                                       test_data._threads_per_stream,
                                                       test_data._core_type,
                                                       false,
                                                       test_data._cpu_pinning,
                                                       test_data._streams_info_table_in};

        ASSERT_EQ(test_data._cpu_pinning, config.get_cpu_pinning());
        ASSERT_EQ(test_data._streams_info_table, config.get_streams_info_table());
        ASSERT_EQ(test_data._stream_processors, config.get_stream_processor_ids());
    }
};

ExecutorConfigTestCase _1sockets_streams_4_threads_1 = {
    // param[in]: proc_type_table, {total processors, number of physical processors, number of Efficient processors,
    // number of hyper threading processors}
    {
        {12, 6, 0, 6, 0, 0},
    },
    // param[in]: cpu_mapping_table, {PROCESSOR_ID, NUMA_ID, SOCKET_ID, CORE_ID, CORE_TYPE, GROUP_ID, Used}
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    4,                                       // param[in]: the number of streams
    1,                                       // param[in]: the number of threads per stream
    ov::hint::SchedulingCoreType::ANY_CORE,  // param[in]: specified cpu core type
    false,                                   // param[in]: specified cpu pinning
    {},                                      // param[in]: streams info table
    // param[out]: streams_info_table, {NUMBER_OF_STREAMS, PROC_TYPE, THREADS_PER_STREAM, STREAM_NUMA_NODE_ID,
    // STREAM_SOCKET_ID}
    {
        {4, MAIN_CORE_PROC, 1, 0, 0},
    },
    // param[out]: stream_processors, the list of processor ids on each stream.
    {},
};

ExecutorConfigTestCase _1sockets_streams_4_threads_0 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    4,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {},
    {},
};

ExecutorConfigTestCase _1sockets_streams_1_threads_12 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    1,
    12,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {1, ALL_PROC, 12, 0, 0},
        {0, MAIN_CORE_PROC, 6, 0, 0},
        {0, HYPER_THREADING_PROC, 6, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _1sockets_streams_1_threads_10 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    1,
    10,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {1, ALL_PROC, 10, 0, 0},
        {0, MAIN_CORE_PROC, 6, 0, 0},
        {0, HYPER_THREADING_PROC, 4, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _1sockets_streams_12_threads_1 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    12,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {6, MAIN_CORE_PROC, 1, 0, 0},
        {6, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _1sockets_streams_13_threads_1 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    13,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {6, MAIN_CORE_PROC, 1, 0, 0},
        {6, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _1sockets_streams_6_threads_1_core_e = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    7,
    1,
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    false,
    {},
    {
        {6, MAIN_CORE_PROC, 1, 0, 0},
        {1, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _1sockets_streams_5_threads_1_binding = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},
        {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},
        {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},
        {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},
        {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},
        {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
    },
    5,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    {},
    {
        {5, MAIN_CORE_PROC, 1, 0, 0},
    },
    {{0}, {2}, {4}, {6}, {8}},
};

ExecutorConfigTestCase _2sockets_streams_36_threads_1 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
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
    36,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {18, MAIN_CORE_PROC, 1, 0, 0},
        {18, MAIN_CORE_PROC, 1, 1, 1},
    },
    {},
};

ExecutorConfigTestCase _2sockets_streams_4_threads_5 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
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
    4,
    5,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {3, MAIN_CORE_PROC, 5, 0, 0},
        {1, MAIN_CORE_PROC, 5, 1, 1},
    },
    {},
};

ExecutorConfigTestCase _2sockets_streams_1_threads_36 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
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
    1,
    36,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {1, ALL_PROC, 36, -1, -1},
        {0, MAIN_CORE_PROC, 18, 0, 0},
        {0, MAIN_CORE_PROC, 18, 1, 1},
    },
    {},
};

ExecutorConfigTestCase _2sockets_streams_1_threads_30 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
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
    1,
    30,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {1, ALL_PROC, 30, -1, -1},
        {0, MAIN_CORE_PROC, 18, 0, 0},
        {0, MAIN_CORE_PROC, 12, 1, 1},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_5_threads_2 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    5,
    2,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {4, MAIN_CORE_PROC, 2, 0, 0},
        {1, EFFICIENT_CORE_PROC, 2, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_5_threads_5 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    5,
    5,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {2, MAIN_CORE_PROC, 4, 0, 0},
        {2, EFFICIENT_CORE_PROC, 4, 0, 0},
        {1, HYPER_THREADING_PROC, 4, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_4_threads_5 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    4,
    5,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {1, MAIN_CORE_PROC, 5, 0, 0},
        {1, EFFICIENT_CORE_PROC, 5, 0, 0},
        {1, HYPER_THREADING_PROC, 5, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_4_threads_1 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    4,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {4, MAIN_CORE_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_5_threads_10 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    5,
    10,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {2, MAIN_CORE_PROC, 4, 0, 0},
        {2, EFFICIENT_CORE_PROC, 4, 0, 0},
        {1, HYPER_THREADING_PROC, 4, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_26_threads_1 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    26,
    1,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {
        {8, MAIN_CORE_PROC, 1, 0, 0},
        {8, EFFICIENT_CORE_PROC, 1, 0, 0},
        {8, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_26_threads_1_p = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    26,
    1,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {},
    {
        {8, MAIN_CORE_PROC, 1, 0, 0},
        {8, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_26_threads_1_e = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    26,
    1,
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    false,
    {},
    {
        {8, EFFICIENT_CORE_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_0 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    0,
    ov::hint::SchedulingCoreType::ANY_CORE,
    false,
    {},
    {},
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_1_p = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    1,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {},
    {
        {1, MAIN_CORE_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_1_e = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    1,
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    false,
    {},
    {
        {1, EFFICIENT_CORE_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_16_p = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    16,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {},
    {
        {1, ALL_PROC, 16, 0, 0},
        {0, MAIN_CORE_PROC, 8, 0, 0},
        {0, HYPER_THREADING_PROC, 8, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_18_p = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    18,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {},
    {
        {1, ALL_PROC, 16, 0, 0},
        {0, MAIN_CORE_PROC, 8, 0, 0},
        {0, HYPER_THREADING_PROC, 8, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_1_threads_10_p = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    10,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {},
    {
        {1, ALL_PROC, 10, 0, 0},
        {0, MAIN_CORE_PROC, 8, 0, 0},
        {0, HYPER_THREADING_PROC, 2, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_10_threads_1_e = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    10,
    1,
    ov::hint::SchedulingCoreType::ECORE_ONLY,
    false,
    {},
    {
        {8, EFFICIENT_CORE_PROC, 1, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_10_threads_1_binding = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    10,
    2,
    ov::hint::SchedulingCoreType::ANY_CORE,
    true,
    {},
    {
        {4, MAIN_CORE_PROC, 2, 0, 0},
        {4, EFFICIENT_CORE_PROC, 2, 0, 0},
        {2, HYPER_THREADING_PROC, 2, 0, 0},
    },
    {{0, 2}, {4, 6}, {8, 10}, {12, 14}, {16, 17}, {18, 19}, {20, 21}, {22, 23}, {1, 3}, {5, 7}},
};

ExecutorConfigTestCase _pecore_streams_info_table_1 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    8,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {
        {2, MAIN_CORE_PROC, 2, 0, 0},
        {2, EFFICIENT_CORE_PROC, 2, 0, 0},
    },
    {
        {2, MAIN_CORE_PROC, 2, 0, 0},
        {2, EFFICIENT_CORE_PROC, 2, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_info_table_2 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    8,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    false,
    {
        {5, MAIN_CORE_PROC, 2, 0, 0},
        {2, EFFICIENT_CORE_PROC, 2, 0, 0},
    },
    {
        {1, MAIN_CORE_PROC, 8, 0, 0},
    },
    {},
};

ExecutorConfigTestCase _pecore_streams_info_table_3 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 12, -1},       {13, 0, 0, 6, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 14, -1},       {15, 0, 0, 7, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 16, -1},  {17, 0, 0, 9, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 11, EFFICIENT_CORE_PROC, 19, -1},
        {20, 0, 0, 12, EFFICIENT_CORE_PROC, 20, -1}, {21, 0, 0, 13, EFFICIENT_CORE_PROC, 21, -1},
        {22, 0, 0, 14, EFFICIENT_CORE_PROC, 22, -1}, {23, 0, 0, 15, EFFICIENT_CORE_PROC, 23, -1},
    },
    1,
    8,
    ov::hint::SchedulingCoreType::PCORE_ONLY,
    true,
    {
        {2, MAIN_CORE_PROC, 2, 0, 0},
        {2, EFFICIENT_CORE_PROC, 2, 0, 0},
        {2, HYPER_THREADING_PROC, 2, 0, 0},
    },
    {
        {2, MAIN_CORE_PROC, 2, 0, 0},
        {2, EFFICIENT_CORE_PROC, 2, 0, 0},
        {2, HYPER_THREADING_PROC, 2, 0, 0},
    },
    {{0, 2}, {4, 6}, {16, 17}, {18, 19}, {1, 3}, {5, 7}},
};

TEST_P(ExecutorConfigTest, ExecutorConfig) {}

INSTANTIATE_TEST_SUITE_P(smoke_ExecutorConfig,
                         ExecutorConfigTest,
                         testing::Values(_1sockets_streams_4_threads_1,
                                         _1sockets_streams_4_threads_0,
                                         _1sockets_streams_1_threads_12,
                                         _1sockets_streams_1_threads_10,
                                         _1sockets_streams_12_threads_1,
                                         _1sockets_streams_13_threads_1,
                                         _1sockets_streams_6_threads_1_core_e,
                                         _1sockets_streams_5_threads_1_binding,
                                         _2sockets_streams_36_threads_1,
                                         _2sockets_streams_4_threads_5,
                                         _2sockets_streams_1_threads_36,
                                         _2sockets_streams_1_threads_30,
                                         _pecore_streams_5_threads_2,
                                         _pecore_streams_5_threads_5,
                                         _pecore_streams_4_threads_5,
                                         _pecore_streams_4_threads_1,
                                         _pecore_streams_5_threads_10,
                                         _pecore_streams_26_threads_1,
                                         _pecore_streams_26_threads_1_p,
                                         _pecore_streams_26_threads_1_e,
                                         _pecore_streams_1_threads_0,
                                         _pecore_streams_1_threads_1_p,
                                         _pecore_streams_1_threads_1_e,
                                         _pecore_streams_1_threads_16_p,
                                         _pecore_streams_1_threads_18_p,
                                         _pecore_streams_1_threads_10_p,
                                         _pecore_streams_10_threads_1_e,
                                         _pecore_streams_10_threads_1_binding,
                                         _pecore_streams_info_table_1,
                                         _pecore_streams_info_table_2,
                                         _pecore_streams_info_table_3));
#endif
}  // namespace
