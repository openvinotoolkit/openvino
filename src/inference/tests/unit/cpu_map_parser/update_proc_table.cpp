// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;

namespace ov {

#ifdef __linux__

struct LinuxSortProcTableTestCase {
    int current_processor_id;
    std::vector<std::vector<int>> _proc_type_table_input;
    std::vector<std::vector<int>> _cpu_mapping_table;
    std::vector<std::vector<int>> _proc_type_table_output;
};

class LinuxSortProcTableTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxSortProcTableTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        CPU& cpu = cpu_info();
        std::vector<std::vector<int>> test_proc_type_table = test_data._proc_type_table_input;

        cpu.sort_table_by_cpu_id(test_data.current_processor_id, test_proc_type_table, test_data._cpu_mapping_table);

        ASSERT_EQ(test_proc_type_table, test_data._proc_type_table_output);
    }
};

LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_1 = {
    2,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 2, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},    {3, 2, 1, 13, HYPER_THREADING_PROC, 13, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {5, 2, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},    {7, 2, 1, 15, HYPER_THREADING_PROC, 15, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {9, 2, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},   {11, 2, 1, 17, HYPER_THREADING_PROC, 17, -1},
        {12, 1, 0, 6, HYPER_THREADING_PROC, 6, -1},   {13, 3, 1, 18, HYPER_THREADING_PROC, 18, -1},
        {14, 1, 0, 7, HYPER_THREADING_PROC, 7, -1},   {15, 3, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {16, 1, 0, 8, HYPER_THREADING_PROC, 8, -1},   {17, 3, 1, 20, HYPER_THREADING_PROC, 20, -1},
        {18, 1, 0, 9, HYPER_THREADING_PROC, 9, -1},   {19, 3, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {20, 1, 0, 10, HYPER_THREADING_PROC, 10, -1}, {21, 3, 1, 22, HYPER_THREADING_PROC, 22, -1},
        {22, 1, 0, 11, HYPER_THREADING_PROC, 11, -1}, {23, 3, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {25, 2, 1, 12, MAIN_CORE_PROC, 12, -1},
        {26, 0, 0, 1, MAIN_CORE_PROC, 1, -1},         {27, 2, 1, 13, MAIN_CORE_PROC, 13, -1},
        {28, 0, 0, 2, MAIN_CORE_PROC, 2, -1},         {29, 2, 1, 14, MAIN_CORE_PROC, 14, -1},
        {30, 0, 0, 3, MAIN_CORE_PROC, 3, -1},         {31, 2, 1, 15, MAIN_CORE_PROC, 15, -1},
        {32, 0, 0, 4, MAIN_CORE_PROC, 4, -1},         {33, 2, 1, 16, MAIN_CORE_PROC, 16, -1},
        {34, 0, 0, 5, MAIN_CORE_PROC, 5, -1},         {35, 2, 1, 17, MAIN_CORE_PROC, 17, -1},
        {36, 1, 0, 6, MAIN_CORE_PROC, 6, -1},         {37, 3, 1, 18, MAIN_CORE_PROC, 18, -1},
        {38, 1, 0, 7, MAIN_CORE_PROC, 7, -1},         {39, 3, 1, 19, MAIN_CORE_PROC, 19, -1},
        {40, 1, 0, 8, MAIN_CORE_PROC, 8, -1},         {41, 3, 1, 20, MAIN_CORE_PROC, 20, -1},
        {42, 1, 0, 9, MAIN_CORE_PROC, 9, -1},         {43, 3, 1, 21, MAIN_CORE_PROC, 21, -1},
        {44, 1, 0, 10, MAIN_CORE_PROC, 10, -1},       {45, 3, 1, 22, MAIN_CORE_PROC, 22, -1},
        {46, 1, 0, 11, MAIN_CORE_PROC, 11, -1},       {47, 3, 1, 23, MAIN_CORE_PROC, 23, -1},
    },
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_2 = {
    16,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 2, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},    {3, 2, 1, 13, HYPER_THREADING_PROC, 13, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {5, 2, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},    {7, 2, 1, 15, HYPER_THREADING_PROC, 15, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {9, 2, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},   {11, 2, 1, 17, HYPER_THREADING_PROC, 17, -1},
        {12, 1, 0, 6, HYPER_THREADING_PROC, 6, -1},   {13, 3, 1, 18, HYPER_THREADING_PROC, 18, -1},
        {14, 1, 0, 7, HYPER_THREADING_PROC, 7, -1},   {15, 3, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {16, 1, 0, 8, HYPER_THREADING_PROC, 8, -1},   {17, 3, 1, 20, HYPER_THREADING_PROC, 20, -1},
        {18, 1, 0, 9, HYPER_THREADING_PROC, 9, -1},   {19, 3, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {20, 1, 0, 10, HYPER_THREADING_PROC, 10, -1}, {21, 3, 1, 22, HYPER_THREADING_PROC, 22, -1},
        {22, 1, 0, 11, HYPER_THREADING_PROC, 11, -1}, {23, 3, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {25, 2, 1, 12, MAIN_CORE_PROC, 12, -1},
        {26, 0, 0, 1, MAIN_CORE_PROC, 1, -1},         {27, 2, 1, 13, MAIN_CORE_PROC, 13, -1},
        {28, 0, 0, 2, MAIN_CORE_PROC, 2, -1},         {29, 2, 1, 14, MAIN_CORE_PROC, 14, -1},
        {30, 0, 0, 3, MAIN_CORE_PROC, 3, -1},         {31, 2, 1, 15, MAIN_CORE_PROC, 15, -1},
        {32, 0, 0, 4, MAIN_CORE_PROC, 4, -1},         {33, 2, 1, 16, MAIN_CORE_PROC, 16, -1},
        {34, 0, 0, 5, MAIN_CORE_PROC, 5, -1},         {35, 2, 1, 17, MAIN_CORE_PROC, 17, -1},
        {36, 1, 0, 6, MAIN_CORE_PROC, 6, -1},         {37, 3, 1, 18, MAIN_CORE_PROC, 18, -1},
        {38, 1, 0, 7, MAIN_CORE_PROC, 7, -1},         {39, 3, 1, 19, MAIN_CORE_PROC, 19, -1},
        {40, 1, 0, 8, MAIN_CORE_PROC, 8, -1},         {41, 3, 1, 20, MAIN_CORE_PROC, 20, -1},
        {42, 1, 0, 9, MAIN_CORE_PROC, 9, -1},         {43, 3, 1, 21, MAIN_CORE_PROC, 21, -1},
        {44, 1, 0, 10, MAIN_CORE_PROC, 10, -1},       {45, 3, 1, 22, MAIN_CORE_PROC, 22, -1},
        {46, 1, 0, 11, MAIN_CORE_PROC, 11, -1},       {47, 3, 1, 23, MAIN_CORE_PROC, 23, -1},
    },
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_3 = {
    7,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 2, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},    {3, 2, 1, 13, HYPER_THREADING_PROC, 13, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {5, 2, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},    {7, 2, 1, 15, HYPER_THREADING_PROC, 15, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {9, 2, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},   {11, 2, 1, 17, HYPER_THREADING_PROC, 17, -1},
        {12, 1, 0, 6, HYPER_THREADING_PROC, 6, -1},   {13, 3, 1, 18, HYPER_THREADING_PROC, 18, -1},
        {14, 1, 0, 7, HYPER_THREADING_PROC, 7, -1},   {15, 3, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {16, 1, 0, 8, HYPER_THREADING_PROC, 8, -1},   {17, 3, 1, 20, HYPER_THREADING_PROC, 20, -1},
        {18, 1, 0, 9, HYPER_THREADING_PROC, 9, -1},   {19, 3, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {20, 1, 0, 10, HYPER_THREADING_PROC, 10, -1}, {21, 3, 1, 22, HYPER_THREADING_PROC, 22, -1},
        {22, 1, 0, 11, HYPER_THREADING_PROC, 11, -1}, {23, 3, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {25, 2, 1, 12, MAIN_CORE_PROC, 12, -1},
        {26, 0, 0, 1, MAIN_CORE_PROC, 1, -1},         {27, 2, 1, 13, MAIN_CORE_PROC, 13, -1},
        {28, 0, 0, 2, MAIN_CORE_PROC, 2, -1},         {29, 2, 1, 14, MAIN_CORE_PROC, 14, -1},
        {30, 0, 0, 3, MAIN_CORE_PROC, 3, -1},         {31, 2, 1, 15, MAIN_CORE_PROC, 15, -1},
        {32, 0, 0, 4, MAIN_CORE_PROC, 4, -1},         {33, 2, 1, 16, MAIN_CORE_PROC, 16, -1},
        {34, 0, 0, 5, MAIN_CORE_PROC, 5, -1},         {35, 2, 1, 17, MAIN_CORE_PROC, 17, -1},
        {36, 1, 0, 6, MAIN_CORE_PROC, 6, -1},         {37, 3, 1, 18, MAIN_CORE_PROC, 18, -1},
        {38, 1, 0, 7, MAIN_CORE_PROC, 7, -1},         {39, 3, 1, 19, MAIN_CORE_PROC, 19, -1},
        {40, 1, 0, 8, MAIN_CORE_PROC, 8, -1},         {41, 3, 1, 20, MAIN_CORE_PROC, 20, -1},
        {42, 1, 0, 9, MAIN_CORE_PROC, 9, -1},         {43, 3, 1, 21, MAIN_CORE_PROC, 21, -1},
        {44, 1, 0, 10, MAIN_CORE_PROC, 10, -1},       {45, 3, 1, 22, MAIN_CORE_PROC, 22, -1},
        {46, 1, 0, 11, MAIN_CORE_PROC, 11, -1},       {47, 3, 1, 23, MAIN_CORE_PROC, 23, -1},
    },
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_4 = {
    21,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 2, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},    {3, 2, 1, 13, HYPER_THREADING_PROC, 13, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},    {5, 2, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},    {7, 2, 1, 15, HYPER_THREADING_PROC, 15, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},    {9, 2, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},   {11, 2, 1, 17, HYPER_THREADING_PROC, 17, -1},
        {12, 1, 0, 6, HYPER_THREADING_PROC, 6, -1},   {13, 3, 1, 18, HYPER_THREADING_PROC, 18, -1},
        {14, 1, 0, 7, HYPER_THREADING_PROC, 7, -1},   {15, 3, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {16, 1, 0, 8, HYPER_THREADING_PROC, 8, -1},   {17, 3, 1, 20, HYPER_THREADING_PROC, 20, -1},
        {18, 1, 0, 9, HYPER_THREADING_PROC, 9, -1},   {19, 3, 1, 21, HYPER_THREADING_PROC, 21, -1},
        {20, 1, 0, 10, HYPER_THREADING_PROC, 10, -1}, {21, 3, 1, 22, HYPER_THREADING_PROC, 22, -1},
        {22, 1, 0, 11, HYPER_THREADING_PROC, 11, -1}, {23, 3, 1, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {25, 2, 1, 12, MAIN_CORE_PROC, 12, -1},
        {26, 0, 0, 1, MAIN_CORE_PROC, 1, -1},         {27, 2, 1, 13, MAIN_CORE_PROC, 13, -1},
        {28, 0, 0, 2, MAIN_CORE_PROC, 2, -1},         {29, 2, 1, 14, MAIN_CORE_PROC, 14, -1},
        {30, 0, 0, 3, MAIN_CORE_PROC, 3, -1},         {31, 2, 1, 15, MAIN_CORE_PROC, 15, -1},
        {32, 0, 0, 4, MAIN_CORE_PROC, 4, -1},         {33, 2, 1, 16, MAIN_CORE_PROC, 16, -1},
        {34, 0, 0, 5, MAIN_CORE_PROC, 5, -1},         {35, 2, 1, 17, MAIN_CORE_PROC, 17, -1},
        {36, 1, 0, 6, MAIN_CORE_PROC, 6, -1},         {37, 3, 1, 18, MAIN_CORE_PROC, 18, -1},
        {38, 1, 0, 7, MAIN_CORE_PROC, 7, -1},         {39, 3, 1, 19, MAIN_CORE_PROC, 19, -1},
        {40, 1, 0, 8, MAIN_CORE_PROC, 8, -1},         {41, 3, 1, 20, MAIN_CORE_PROC, 20, -1},
        {42, 1, 0, 9, MAIN_CORE_PROC, 9, -1},         {43, 3, 1, 21, MAIN_CORE_PROC, 21, -1},
        {44, 1, 0, 10, MAIN_CORE_PROC, 10, -1},       {45, 3, 1, 22, MAIN_CORE_PROC, 22, -1},
        {46, 1, 0, 11, MAIN_CORE_PROC, 11, -1},       {47, 3, 1, 23, MAIN_CORE_PROC, 23, -1},
    },
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}},
};

TEST_P(LinuxSortProcTableTests, LinuxProcTable) {}

INSTANTIATE_TEST_SUITE_P(CPUMap,
                         LinuxSortProcTableTests,
                         testing::Values(proc_table_2sockets_24cores_hyperthreading_1,
                                         proc_table_2sockets_24cores_hyperthreading_2,
                                         proc_table_2sockets_24cores_hyperthreading_3,
                                         proc_table_2sockets_24cores_hyperthreading_4));
#endif
}  // namespace ov
