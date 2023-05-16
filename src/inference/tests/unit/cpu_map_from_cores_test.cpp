// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "ie_system_conf.h"
#include "streams_executor.hpp"

using namespace testing;
using namespace ov;

namespace {

#ifdef __linux__

struct LinuxCpuMapTestCase {
    int _processors;
    int _sockets;
    int _cores;
    int _phys_cores;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
};

class LinuxGetCpuMapFromCoresTests : public CommonTestUtils::TestsCommon,
                                     public testing::WithParamInterface<std::tuple<LinuxCpuMapTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_proc_type_table;
        std::vector<std::vector<int>> test_cpu_mapping_table;

        ov::get_cpu_mapping_from_cores(test_data._processors,
                                       test_data._sockets,
                                       test_data._cores,
                                       test_data._phys_cores,
                                       test_proc_type_table,
                                       test_cpu_mapping_table);

        ASSERT_EQ(test_data._proc_type_table, test_proc_type_table);
        ASSERT_EQ(test_data._cpu_mapping_table, test_cpu_mapping_table);
    }
};

LinuxCpuMapTestCase _2sockets_104cores_hyperthreading = {
    208,
    2,
    104,
    104,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {
        {0, 0, 0, HYPER_THREADING_PROC, 0, -1},       {1, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 2, HYPER_THREADING_PROC, 2, -1},       {3, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 4, HYPER_THREADING_PROC, 4, -1},       {5, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 6, HYPER_THREADING_PROC, 6, -1},       {7, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 8, HYPER_THREADING_PROC, 8, -1},       {9, 0, 9, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 10, HYPER_THREADING_PROC, 10, -1},    {11, 0, 11, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 12, HYPER_THREADING_PROC, 12, -1},    {13, 0, 13, HYPER_THREADING_PROC, 13, -1},
        {14, 0, 14, HYPER_THREADING_PROC, 14, -1},    {15, 0, 15, HYPER_THREADING_PROC, 15, -1},
        {16, 0, 16, HYPER_THREADING_PROC, 16, -1},    {17, 0, 17, HYPER_THREADING_PROC, 17, -1},
        {18, 0, 18, HYPER_THREADING_PROC, 18, -1},    {19, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 20, HYPER_THREADING_PROC, 20, -1},    {21, 0, 21, HYPER_THREADING_PROC, 21, -1},
        {22, 0, 22, HYPER_THREADING_PROC, 22, -1},    {23, 0, 23, HYPER_THREADING_PROC, 23, -1},
        {24, 0, 24, HYPER_THREADING_PROC, 24, -1},    {25, 0, 25, HYPER_THREADING_PROC, 25, -1},
        {26, 0, 26, HYPER_THREADING_PROC, 26, -1},    {27, 0, 27, HYPER_THREADING_PROC, 27, -1},
        {28, 0, 28, HYPER_THREADING_PROC, 28, -1},    {29, 0, 29, HYPER_THREADING_PROC, 29, -1},
        {30, 0, 30, HYPER_THREADING_PROC, 30, -1},    {31, 0, 31, HYPER_THREADING_PROC, 31, -1},
        {32, 0, 32, HYPER_THREADING_PROC, 32, -1},    {33, 0, 33, HYPER_THREADING_PROC, 33, -1},
        {34, 0, 34, HYPER_THREADING_PROC, 34, -1},    {35, 0, 35, HYPER_THREADING_PROC, 35, -1},
        {36, 0, 36, HYPER_THREADING_PROC, 36, -1},    {37, 0, 37, HYPER_THREADING_PROC, 37, -1},
        {38, 0, 38, HYPER_THREADING_PROC, 38, -1},    {39, 0, 39, HYPER_THREADING_PROC, 39, -1},
        {40, 0, 40, HYPER_THREADING_PROC, 40, -1},    {41, 0, 41, HYPER_THREADING_PROC, 41, -1},
        {42, 0, 42, HYPER_THREADING_PROC, 42, -1},    {43, 0, 43, HYPER_THREADING_PROC, 43, -1},
        {44, 0, 44, HYPER_THREADING_PROC, 44, -1},    {45, 0, 45, HYPER_THREADING_PROC, 45, -1},
        {46, 0, 46, HYPER_THREADING_PROC, 46, -1},    {47, 0, 47, HYPER_THREADING_PROC, 47, -1},
        {48, 0, 48, HYPER_THREADING_PROC, 48, -1},    {49, 0, 49, HYPER_THREADING_PROC, 49, -1},
        {50, 0, 50, HYPER_THREADING_PROC, 50, -1},    {51, 0, 51, HYPER_THREADING_PROC, 51, -1},
        {52, 1, 52, HYPER_THREADING_PROC, 52, -1},    {53, 1, 53, HYPER_THREADING_PROC, 53, -1},
        {54, 1, 54, HYPER_THREADING_PROC, 54, -1},    {55, 1, 55, HYPER_THREADING_PROC, 55, -1},
        {56, 1, 56, HYPER_THREADING_PROC, 56, -1},    {57, 1, 57, HYPER_THREADING_PROC, 57, -1},
        {58, 1, 58, HYPER_THREADING_PROC, 58, -1},    {59, 1, 59, HYPER_THREADING_PROC, 59, -1},
        {60, 1, 60, HYPER_THREADING_PROC, 60, -1},    {61, 1, 61, HYPER_THREADING_PROC, 61, -1},
        {62, 1, 62, HYPER_THREADING_PROC, 62, -1},    {63, 1, 63, HYPER_THREADING_PROC, 63, -1},
        {64, 1, 64, HYPER_THREADING_PROC, 64, -1},    {65, 1, 65, HYPER_THREADING_PROC, 65, -1},
        {66, 1, 66, HYPER_THREADING_PROC, 66, -1},    {67, 1, 67, HYPER_THREADING_PROC, 67, -1},
        {68, 1, 68, HYPER_THREADING_PROC, 68, -1},    {69, 1, 69, HYPER_THREADING_PROC, 69, -1},
        {70, 1, 70, HYPER_THREADING_PROC, 70, -1},    {71, 1, 71, HYPER_THREADING_PROC, 71, -1},
        {72, 1, 72, HYPER_THREADING_PROC, 72, -1},    {73, 1, 73, HYPER_THREADING_PROC, 73, -1},
        {74, 1, 74, HYPER_THREADING_PROC, 74, -1},    {75, 1, 75, HYPER_THREADING_PROC, 75, -1},
        {76, 1, 76, HYPER_THREADING_PROC, 76, -1},    {77, 1, 77, HYPER_THREADING_PROC, 77, -1},
        {78, 1, 78, HYPER_THREADING_PROC, 78, -1},    {79, 1, 79, HYPER_THREADING_PROC, 79, -1},
        {80, 1, 80, HYPER_THREADING_PROC, 80, -1},    {81, 1, 81, HYPER_THREADING_PROC, 81, -1},
        {82, 1, 82, HYPER_THREADING_PROC, 82, -1},    {83, 1, 83, HYPER_THREADING_PROC, 83, -1},
        {84, 1, 84, HYPER_THREADING_PROC, 84, -1},    {85, 1, 85, HYPER_THREADING_PROC, 85, -1},
        {86, 1, 86, HYPER_THREADING_PROC, 86, -1},    {87, 1, 87, HYPER_THREADING_PROC, 87, -1},
        {88, 1, 88, HYPER_THREADING_PROC, 88, -1},    {89, 1, 89, HYPER_THREADING_PROC, 89, -1},
        {90, 1, 90, HYPER_THREADING_PROC, 90, -1},    {91, 1, 91, HYPER_THREADING_PROC, 91, -1},
        {92, 1, 92, HYPER_THREADING_PROC, 92, -1},    {93, 1, 93, HYPER_THREADING_PROC, 93, -1},
        {94, 1, 94, HYPER_THREADING_PROC, 94, -1},    {95, 1, 95, HYPER_THREADING_PROC, 95, -1},
        {96, 1, 96, HYPER_THREADING_PROC, 96, -1},    {97, 1, 97, HYPER_THREADING_PROC, 97, -1},
        {98, 1, 98, HYPER_THREADING_PROC, 98, -1},    {99, 1, 99, HYPER_THREADING_PROC, 99, -1},
        {100, 1, 100, HYPER_THREADING_PROC, 100, -1}, {101, 1, 101, HYPER_THREADING_PROC, 101, -1},
        {102, 1, 102, HYPER_THREADING_PROC, 102, -1}, {103, 1, 103, HYPER_THREADING_PROC, 103, -1},
        {104, 0, 0, MAIN_CORE_PROC, 0, -1},           {105, 0, 1, MAIN_CORE_PROC, 1, -1},
        {106, 0, 2, MAIN_CORE_PROC, 2, -1},           {107, 0, 3, MAIN_CORE_PROC, 3, -1},
        {108, 0, 4, MAIN_CORE_PROC, 4, -1},           {109, 0, 5, MAIN_CORE_PROC, 5, -1},
        {110, 0, 6, MAIN_CORE_PROC, 6, -1},           {111, 0, 7, MAIN_CORE_PROC, 7, -1},
        {112, 0, 8, MAIN_CORE_PROC, 8, -1},           {113, 0, 9, MAIN_CORE_PROC, 9, -1},
        {114, 0, 10, MAIN_CORE_PROC, 10, -1},         {115, 0, 11, MAIN_CORE_PROC, 11, -1},
        {116, 0, 12, MAIN_CORE_PROC, 12, -1},         {117, 0, 13, MAIN_CORE_PROC, 13, -1},
        {118, 0, 14, MAIN_CORE_PROC, 14, -1},         {119, 0, 15, MAIN_CORE_PROC, 15, -1},
        {120, 0, 16, MAIN_CORE_PROC, 16, -1},         {121, 0, 17, MAIN_CORE_PROC, 17, -1},
        {122, 0, 18, MAIN_CORE_PROC, 18, -1},         {123, 0, 19, MAIN_CORE_PROC, 19, -1},
        {124, 0, 20, MAIN_CORE_PROC, 20, -1},         {125, 0, 21, MAIN_CORE_PROC, 21, -1},
        {126, 0, 22, MAIN_CORE_PROC, 22, -1},         {127, 0, 23, MAIN_CORE_PROC, 23, -1},
        {128, 0, 24, MAIN_CORE_PROC, 24, -1},         {129, 0, 25, MAIN_CORE_PROC, 25, -1},
        {130, 0, 26, MAIN_CORE_PROC, 26, -1},         {131, 0, 27, MAIN_CORE_PROC, 27, -1},
        {132, 0, 28, MAIN_CORE_PROC, 28, -1},         {133, 0, 29, MAIN_CORE_PROC, 29, -1},
        {134, 0, 30, MAIN_CORE_PROC, 30, -1},         {135, 0, 31, MAIN_CORE_PROC, 31, -1},
        {136, 0, 32, MAIN_CORE_PROC, 32, -1},         {137, 0, 33, MAIN_CORE_PROC, 33, -1},
        {138, 0, 34, MAIN_CORE_PROC, 34, -1},         {139, 0, 35, MAIN_CORE_PROC, 35, -1},
        {140, 0, 36, MAIN_CORE_PROC, 36, -1},         {141, 0, 37, MAIN_CORE_PROC, 37, -1},
        {142, 0, 38, MAIN_CORE_PROC, 38, -1},         {143, 0, 39, MAIN_CORE_PROC, 39, -1},
        {144, 0, 40, MAIN_CORE_PROC, 40, -1},         {145, 0, 41, MAIN_CORE_PROC, 41, -1},
        {146, 0, 42, MAIN_CORE_PROC, 42, -1},         {147, 0, 43, MAIN_CORE_PROC, 43, -1},
        {148, 0, 44, MAIN_CORE_PROC, 44, -1},         {149, 0, 45, MAIN_CORE_PROC, 45, -1},
        {150, 0, 46, MAIN_CORE_PROC, 46, -1},         {151, 0, 47, MAIN_CORE_PROC, 47, -1},
        {152, 0, 48, MAIN_CORE_PROC, 48, -1},         {153, 0, 49, MAIN_CORE_PROC, 49, -1},
        {154, 0, 50, MAIN_CORE_PROC, 50, -1},         {155, 0, 51, MAIN_CORE_PROC, 51, -1},
        {156, 1, 52, MAIN_CORE_PROC, 52, -1},         {157, 1, 53, MAIN_CORE_PROC, 53, -1},
        {158, 1, 54, MAIN_CORE_PROC, 54, -1},         {159, 1, 55, MAIN_CORE_PROC, 55, -1},
        {160, 1, 56, MAIN_CORE_PROC, 56, -1},         {161, 1, 57, MAIN_CORE_PROC, 57, -1},
        {162, 1, 58, MAIN_CORE_PROC, 58, -1},         {163, 1, 59, MAIN_CORE_PROC, 59, -1},
        {164, 1, 60, MAIN_CORE_PROC, 60, -1},         {165, 1, 61, MAIN_CORE_PROC, 61, -1},
        {166, 1, 62, MAIN_CORE_PROC, 62, -1},         {167, 1, 63, MAIN_CORE_PROC, 63, -1},
        {168, 1, 64, MAIN_CORE_PROC, 64, -1},         {169, 1, 65, MAIN_CORE_PROC, 65, -1},
        {170, 1, 66, MAIN_CORE_PROC, 66, -1},         {171, 1, 67, MAIN_CORE_PROC, 67, -1},
        {172, 1, 68, MAIN_CORE_PROC, 68, -1},         {173, 1, 69, MAIN_CORE_PROC, 69, -1},
        {174, 1, 70, MAIN_CORE_PROC, 70, -1},         {175, 1, 71, MAIN_CORE_PROC, 71, -1},
        {176, 1, 72, MAIN_CORE_PROC, 72, -1},         {177, 1, 73, MAIN_CORE_PROC, 73, -1},
        {178, 1, 74, MAIN_CORE_PROC, 74, -1},         {179, 1, 75, MAIN_CORE_PROC, 75, -1},
        {180, 1, 76, MAIN_CORE_PROC, 76, -1},         {181, 1, 77, MAIN_CORE_PROC, 77, -1},
        {182, 1, 78, MAIN_CORE_PROC, 78, -1},         {183, 1, 79, MAIN_CORE_PROC, 79, -1},
        {184, 1, 80, MAIN_CORE_PROC, 80, -1},         {185, 1, 81, MAIN_CORE_PROC, 81, -1},
        {186, 1, 82, MAIN_CORE_PROC, 82, -1},         {187, 1, 83, MAIN_CORE_PROC, 83, -1},
        {188, 1, 84, MAIN_CORE_PROC, 84, -1},         {189, 1, 85, MAIN_CORE_PROC, 85, -1},
        {190, 1, 86, MAIN_CORE_PROC, 86, -1},         {191, 1, 87, MAIN_CORE_PROC, 87, -1},
        {192, 1, 88, MAIN_CORE_PROC, 88, -1},         {193, 1, 89, MAIN_CORE_PROC, 89, -1},
        {194, 1, 90, MAIN_CORE_PROC, 90, -1},         {195, 1, 91, MAIN_CORE_PROC, 91, -1},
        {196, 1, 92, MAIN_CORE_PROC, 92, -1},         {197, 1, 93, MAIN_CORE_PROC, 93, -1},
        {198, 1, 94, MAIN_CORE_PROC, 94, -1},         {199, 1, 95, MAIN_CORE_PROC, 95, -1},
        {200, 1, 96, MAIN_CORE_PROC, 96, -1},         {201, 1, 97, MAIN_CORE_PROC, 97, -1},
        {202, 1, 98, MAIN_CORE_PROC, 98, -1},         {203, 1, 99, MAIN_CORE_PROC, 99, -1},
        {204, 1, 100, MAIN_CORE_PROC, 100, -1},       {205, 1, 101, MAIN_CORE_PROC, 101, -1},
        {206, 1, 102, MAIN_CORE_PROC, 102, -1},       {207, 1, 103, MAIN_CORE_PROC, 103, -1},
    },
};
LinuxCpuMapTestCase _2sockets_48cores = {
    48,
    2,
    48,
    48,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {
        {0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 1, MAIN_CORE_PROC, 1, -1},    {2, 0, 2, MAIN_CORE_PROC, 2, -1},
        {3, 0, 3, MAIN_CORE_PROC, 3, -1},    {4, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 7, MAIN_CORE_PROC, 7, -1},    {8, 0, 8, MAIN_CORE_PROC, 8, -1},
        {9, 0, 9, MAIN_CORE_PROC, 9, -1},    {10, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 13, MAIN_CORE_PROC, 13, -1}, {14, 0, 14, MAIN_CORE_PROC, 14, -1},
        {15, 0, 15, MAIN_CORE_PROC, 15, -1}, {16, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 19, MAIN_CORE_PROC, 19, -1}, {20, 0, 20, MAIN_CORE_PROC, 20, -1},
        {21, 0, 21, MAIN_CORE_PROC, 21, -1}, {22, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 25, MAIN_CORE_PROC, 25, -1}, {26, 1, 26, MAIN_CORE_PROC, 26, -1},
        {27, 1, 27, MAIN_CORE_PROC, 27, -1}, {28, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 31, MAIN_CORE_PROC, 31, -1}, {32, 1, 32, MAIN_CORE_PROC, 32, -1},
        {33, 1, 33, MAIN_CORE_PROC, 33, -1}, {34, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 37, MAIN_CORE_PROC, 37, -1}, {38, 1, 38, MAIN_CORE_PROC, 38, -1},
        {39, 1, 39, MAIN_CORE_PROC, 39, -1}, {40, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 43, MAIN_CORE_PROC, 43, -1}, {44, 1, 44, MAIN_CORE_PROC, 44, -1},
        {45, 1, 45, MAIN_CORE_PROC, 45, -1}, {46, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 47, MAIN_CORE_PROC, 47, -1},
    }
};
LinuxCpuMapTestCase _2sockets_20cores_hyperthreading = {
    40,
    2,
    20,
    20,
    {{40, 20, 0, 20}, {20, 10, 0, 10}, {20, 10, 0, 10}},
    {
        {0, 0, 0, HYPER_THREADING_PROC, 0, -1},    {1, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 2, HYPER_THREADING_PROC, 2, -1},    {3, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 4, HYPER_THREADING_PROC, 4, -1},    {5, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 6, HYPER_THREADING_PROC, 6, -1},    {7, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 8, HYPER_THREADING_PROC, 8, -1},    {9, 0, 9, HYPER_THREADING_PROC, 9, -1},
        {10, 1, 10, HYPER_THREADING_PROC, 10, -1}, {11, 1, 11, HYPER_THREADING_PROC, 11, -1},
        {12, 1, 12, HYPER_THREADING_PROC, 12, -1}, {13, 1, 13, HYPER_THREADING_PROC, 13, -1},
        {14, 1, 14, HYPER_THREADING_PROC, 14, -1}, {15, 1, 15, HYPER_THREADING_PROC, 15, -1},
        {16, 1, 16, HYPER_THREADING_PROC, 16, -1}, {17, 1, 17, HYPER_THREADING_PROC, 17, -1},
        {18, 1, 18, HYPER_THREADING_PROC, 18, -1}, {19, 1, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, MAIN_CORE_PROC, 0, -1},         {21, 0, 1, MAIN_CORE_PROC, 1, -1},
        {22, 0, 2, MAIN_CORE_PROC, 2, -1},         {23, 0, 3, MAIN_CORE_PROC, 3, -1},
        {24, 0, 4, MAIN_CORE_PROC, 4, -1},         {25, 0, 5, MAIN_CORE_PROC, 5, -1},
        {26, 0, 6, MAIN_CORE_PROC, 6, -1},         {27, 0, 7, MAIN_CORE_PROC, 7, -1},
        {28, 0, 8, MAIN_CORE_PROC, 8, -1},         {29, 0, 9, MAIN_CORE_PROC, 9, -1},
        {30, 1, 10, MAIN_CORE_PROC, 10, -1},       {31, 1, 11, MAIN_CORE_PROC, 11, -1},
        {32, 1, 12, MAIN_CORE_PROC, 12, -1},       {33, 1, 13, MAIN_CORE_PROC, 13, -1},
        {34, 1, 14, MAIN_CORE_PROC, 14, -1},       {35, 1, 15, MAIN_CORE_PROC, 15, -1},
        {36, 1, 16, MAIN_CORE_PROC, 16, -1},       {37, 1, 17, MAIN_CORE_PROC, 17, -1},
        {38, 1, 18, MAIN_CORE_PROC, 18, -1},       {39, 1, 19, MAIN_CORE_PROC, 19, -1},
    }
};
LinuxCpuMapTestCase _1sockets_14cores_hyperthreading = {
    20,
    1,
    14,
    6,
    {{20, 6, 8, 6}},
    {
        {0, 0, 0, HYPER_THREADING_PROC, 0, -1},  {1, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 1, HYPER_THREADING_PROC, 1, -1},  {3, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 2, HYPER_THREADING_PROC, 2, -1},  {5, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 3, HYPER_THREADING_PROC, 3, -1},  {7, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 4, HYPER_THREADING_PROC, 4, -1},  {9, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 5, HYPER_THREADING_PROC, 5, -1}, {11, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 6, EFFICIENT_CORE_PROC, 6, -1},  {13, 0, 7, EFFICIENT_CORE_PROC, 6, -1},
        {14, 0, 8, EFFICIENT_CORE_PROC, 6, -1},  {15, 0, 9, EFFICIENT_CORE_PROC, 6, -1},
        {16, 0, 10, EFFICIENT_CORE_PROC, 7, -1}, {17, 0, 11, EFFICIENT_CORE_PROC, 7, -1},
        {18, 0, 12, EFFICIENT_CORE_PROC, 7, -1}, {19, 0, 13, EFFICIENT_CORE_PROC, 7, -1},
    }
};
LinuxCpuMapTestCase _1sockets_10cores_hyperthreading{
    12,
    1,
    10,
    2,
    {{12, 2, 8, 2}},
    {
        {0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {3, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 2, EFFICIENT_CORE_PROC, 2, -1},
        {5, 0, 3, EFFICIENT_CORE_PROC, 2, -1},
        {6, 0, 4, EFFICIENT_CORE_PROC, 2, -1},
        {7, 0, 5, EFFICIENT_CORE_PROC, 2, -1},
        {8, 0, 6, EFFICIENT_CORE_PROC, 3, -1},
        {9, 0, 7, EFFICIENT_CORE_PROC, 3, -1},
        {10, 0, 8, EFFICIENT_CORE_PROC, 3, -1},
        {11, 0, 9, EFFICIENT_CORE_PROC, 3, -1},
    }
};
LinuxCpuMapTestCase _1sockets_8cores_hyperthreading = {
    12,
    1,
    8,
    4,
    {{12, 4, 4, 4}},
    {
        {0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {3, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {5, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {7, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 4, EFFICIENT_CORE_PROC, 4, -1},
        {9, 0, 5, EFFICIENT_CORE_PROC, 4, -1},
        {10, 0, 6, EFFICIENT_CORE_PROC, 4, -1},
        {11, 0, 7, EFFICIENT_CORE_PROC, 4, -1},
    }
};

TEST_P(LinuxGetCpuMapFromCoresTests, LinuxCpuMap) {}

INSTANTIATE_TEST_SUITE_P(CPUMap,
                         LinuxGetCpuMapFromCoresTests,
                         testing::Values(_2sockets_104cores_hyperthreading,
                                         _2sockets_48cores,
                                         _2sockets_20cores_hyperthreading,
                                         _1sockets_14cores_hyperthreading,
                                         _1sockets_10cores_hyperthreading,
                                         _1sockets_8cores_hyperthreading));

#endif

}  // namespace
