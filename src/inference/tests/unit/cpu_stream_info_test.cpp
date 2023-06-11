// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "ie_system_conf.h"
#include "openvino/runtime/threading/istreams_executor.hpp"

using namespace testing;
using namespace ov;
using namespace threading;

namespace {

#if defined(__linux__) || defined(_WIN32)

struct LinuxCpuStreamTypeCase {
    int _stream_id;
    bool _cpu_reservation;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _streams_info_table;
    std::vector<int> _stream_numa_node_ids;
    StreamCreateType _stream_type;
    int _concurrency;
    int _core_type;
    int _numa_node_id;
};

class LinuxCpuStreamTypeTests : public CommonTestUtils::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxCpuStreamTypeCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        StreamCreateType test_stream_type;
        int test_concurrency;
        int test_core_type;
        int test_numa_node_id;

        get_cur_stream_info(test_data._stream_id,
                            test_data._cpu_reservation,
                            test_data._proc_type_table,
                            test_data._streams_info_table,
                            test_data._stream_numa_node_ids,
                            test_stream_type,
                            test_concurrency,
                            test_core_type,
                            test_numa_node_id);

        ASSERT_EQ(test_data._stream_type, test_stream_type);
        ASSERT_EQ(test_data._concurrency, test_concurrency);
        ASSERT_EQ(test_data._core_type, test_core_type);
        ASSERT_EQ(test_data._numa_node_id, test_numa_node_id);
    }
};

LinuxCpuStreamTypeCase _2sockets_72cores_nobinding_numanode0 = {
    18,
    false,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{36, MAIN_CORE_PROC, 1}},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    STREAM_WITH_NUMA_ID,
    1,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _2sockets_72cores_nobinding_numanode1 = {
    1,
    false,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{36, MAIN_CORE_PROC, 1}},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    STREAM_WITH_NUMA_ID,
    1,
    MAIN_CORE_PROC,
    1,
};
LinuxCpuStreamTypeCase _2sockets_72cores_nobinding_numanode_all = {
    8,
    false,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{9, MAIN_CORE_PROC, 4}},
    {1, 1, 1, 1, 0, 0, 0, 0, -1},
    STREAM_WITH_NUMA_ID,
    4,
    MAIN_CORE_PROC,
    -1,
};
LinuxCpuStreamTypeCase _2sockets_72cores_binding_numanode_0 = {
    4,
    true,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{9, MAIN_CORE_PROC, 4}},
    {1, 1, 1, 1, 0, 0, 0, 0, -1},
    STREAM_WITH_OBSERVE,
    4,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _2sockets_72cores_binding_numanode_1 = {
    0,
    true,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{9, MAIN_CORE_PROC, 4}},
    {1, 1, 1, 1, 0, 0, 0, 0, -1},
    STREAM_WITH_OBSERVE,
    4,
    MAIN_CORE_PROC,
    1,
};
LinuxCpuStreamTypeCase _2sockets_72cores_binding_numanode_all = {
    8,
    true,
    {{72, 36, 0, 36}, {36, 18, 0, 18}, {36, 18, 0, 18}},
    {{9, MAIN_CORE_PROC, 4}},
    {1, 1, 1, 1, 0, 0, 0, 0, -1},
    STREAM_WITH_OBSERVE,
    4,
    MAIN_CORE_PROC,
    -1,
};
LinuxCpuStreamTypeCase _1sockets_4cores_nobinding = {
    0,
    false,
    {{8, 4, 0, 4}},
    {{1, MAIN_CORE_PROC, 8}},
    {0},
    STREAM_WITHOUT_PARAM,
    8,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_4cores_binding = {
    0,
    true,
    {{8, 4, 0, 4}},
    {{4, MAIN_CORE_PROC, 1}},
    {0, 0, 0, 0},
    STREAM_WITH_OBSERVE,
    1,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_12cores_pcore_nobinding = {
    0,
    false,
    {{20, 8, 4, 8}},
    {{1, MAIN_CORE_PROC, 8}},
    {0},
    STREAM_WITH_CORE_TYPE,
    8,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_12cores_pcore_binding = {
    0,
    true,
    {{20, 8, 4, 8}},
    {{2, MAIN_CORE_PROC, 4}},
    {0, 0},
#    if defined(__linux__)
    STREAM_WITH_OBSERVE,
#    else
    STREAM_WITH_CORE_TYPE,
#    endif
    4,
    MAIN_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_12cores_ecore_nobinding = {
    0,
    false,
    {{20, 8, 4, 8}},
    {{2, EFFICIENT_CORE_PROC, 2}},
    {0, 0},
    STREAM_WITH_CORE_TYPE,
    2,
    EFFICIENT_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_12cores_ecore_binding = {
    0,
    true,
    {{20, 8, 4, 8}},
    {{4, EFFICIENT_CORE_PROC, 1}},
    {0, 0, 0, 0},
#    if defined(__linux__)
    STREAM_WITH_OBSERVE,
#    else
    STREAM_WITH_CORE_TYPE,
#    endif
    1,
    EFFICIENT_CORE_PROC,
    0,
};
LinuxCpuStreamTypeCase _1sockets_24cores_all_proc = {
    0,
    false,
    {{32, 8, 16, 8}},
    {{1, ALL_PROC, 24}},
    {0},
    STREAM_WITHOUT_PARAM,
    24,
    ALL_PROC,
    0,
};

TEST_P(LinuxCpuStreamTypeTests, LinuxCpuStreamType) {}

INSTANTIATE_TEST_SUITE_P(CpuStreamType,
                         LinuxCpuStreamTypeTests,
                         testing::Values(_2sockets_72cores_nobinding_numanode0,
                                         _2sockets_72cores_nobinding_numanode1,
                                         _2sockets_72cores_binding_numanode_all,
                                         _2sockets_72cores_binding_numanode_0,
                                         _2sockets_72cores_binding_numanode_1,
                                         _2sockets_72cores_binding_numanode_all,
                                         _1sockets_4cores_nobinding,
                                         _1sockets_4cores_binding,
                                         _1sockets_12cores_pcore_nobinding,
                                         _1sockets_12cores_pcore_binding,
                                         _1sockets_12cores_ecore_nobinding,
                                         _1sockets_12cores_ecore_binding,
                                         _1sockets_24cores_all_proc));
#endif
}  // namespace
