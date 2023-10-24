// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

// #include "ie_system_conf.h"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;
using namespace threading;

namespace {

#if defined(__linux__) || defined(_WIN32)

struct UpdateExecutorConfigTestCase {
    ov::threading::IStreamsExecutor::Config _config;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
    int _num_streams;
    int _threads_per_stream;
    ov::threading::IStreamsExecutor::Config::PreferredCoreType _core_type;
    bool _cpu_pinning;
    std::vector<std::vector<int>> _streams_info_table;
    std::vector<std::vector<int>> _stream_processors;
};

class UpdateExecutorConfigTest : public ov::test::TestsCommon,
                                 public testing::WithParamInterface<std::tuple<UpdateExecutorConfigTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        CPU& cpu = cpu_info();
        cpu._org_proc_type_table = test_data._proc_type_table;
        cpu._proc_type_table = test_data._proc_type_table;
        cpu._cpu_mapping_table = test_data._cpu_mapping_table;
        cpu._numa_nodes = 1;

        ov::threading::IStreamsExecutor::Config::update_executor_config(test_data._config,
                                                                        test_data._num_streams,
                                                                        test_data._threads_per_stream,
                                                                        test_data._core_type,
                                                                        test_data._cpu_pinning);

        ASSERT_EQ(test_data._num_streams, test_data._config._streams);
        ASSERT_EQ(test_data._threads_per_stream, test_data._config._threadsPerStream);
        ASSERT_EQ(test_data._core_type, test_data._config._threadPreferredCoreType);
        ASSERT_EQ(test_data._cpu_pinning, test_data._config._cpu_reservation);
        ASSERT_EQ(test_data._num_streams, test_data._config._streams);
        ASSERT_EQ(test_data._streams_info_table, test_data._config._streams_info_table);
        ASSERT_EQ(test_data._stream_processors, test_data._config._stream_processor_ids);
    }
};

UpdateExecutorConfigTestCase _update_num_streams = {
    ov::threading::IStreamsExecutor::Config{"update num streams test"},  // param[in]: initial configuration
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
    4,                                             // param[in]: the number of streams
    1,                                             // param[in]: the number of threads per stream
    ov::threading::IStreamsExecutor::Config::ANY,  // param[in]: specified cpu core type
    false,                                         // param[in]: specified cpu pinning
    // param[out]: streams_info_table, {NUMBER_OF_STREAMS, PROC_TYPE, THREADS_PER_STREAM, STREAM_NUMA_NODE_ID,
    // STREAM_SOCKET_ID}
    {
        {4, MAIN_CORE_PROC, 1, 0, 0},
    },
    // param[out]: stream_processors, the list of processor ids on each stream.
    {},
};

UpdateExecutorConfigTestCase _update_core_type = {
    ov::threading::IStreamsExecutor::Config{"update core type test"},
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
    8,
    1,
    ov::threading::IStreamsExecutor::Config::LITTLE,
    false,
    {
        {8, EFFICIENT_CORE_PROC, 1, 0, 0},
    },
    {},
};

UpdateExecutorConfigTestCase _update_cpu_pinning = {
    ov::threading::IStreamsExecutor::Config{"update cpu pinning test"},
    {
        {8, 4, 0, 4, 0, 0},
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
    },
    8,
    1,
    ov::threading::IStreamsExecutor::Config::ANY,
    true,
    {
        {4, MAIN_CORE_PROC, 1, 0, 0},
        {4, HYPER_THREADING_PROC, 1, 0, 0},
    },
    {
        {0},
        {2},
        {4},
        {6},
        {1},
        {3},
        {5},
        {7},
    },
};

TEST_P(UpdateExecutorConfigTest, UpdateExecutorConfig) {}

INSTANTIATE_TEST_SUITE_P(smoke_UpdateExecutorConfig,
                         UpdateExecutorConfigTest,
                         testing::Values(_update_num_streams, _update_core_type, _update_cpu_pinning));
#endif
}  // namespace
