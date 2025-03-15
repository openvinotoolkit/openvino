// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_map_scheduling.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;

namespace {

struct CpuPinningTestCase {
    bool input_cpu_pinning;
    bool input_changed;
    std::vector<std::vector<int>> input_cpu_map_table;
    std::vector<std::vector<int>> input_proc_type_table;
    std::vector<std::vector<int>> input_stream_info_table;
    bool output_cpu_pinning;
};

class CpuPinningTests : public ov::test::TestsCommon,
                        public testing::WithParamInterface<std::tuple<CpuPinningTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());
        CPU& cpu = cpu_info();
        cpu._cpu_mapping_table = test_data.input_cpu_map_table;
        cpu._proc_type_table = test_data.input_proc_type_table;
        cpu._org_proc_type_table = test_data.input_proc_type_table;
        cpu._numa_nodes = cpu._proc_type_table.size() > 1 ? static_cast<int>(cpu._proc_type_table.size()) - 1 : 1;
        cpu._sockets = cpu._numa_nodes;
        bool cpu_reservation = false;

        test_data.input_cpu_pinning = ov::intel_cpu::check_cpu_pinning(test_data.input_cpu_pinning,
                                                                       test_data.input_changed,
                                                                       cpu_reservation,
                                                                       test_data.input_stream_info_table);

        ASSERT_EQ(test_data.output_cpu_pinning, test_data.input_cpu_pinning);
    }
};

TEST_P(CpuPinningTests, CpuPinning) {}

CpuPinningTestCase cpu_pinning_macos_mock_set_true = {
    true,                             // param[in]: simulated settting for cpu pinning property
    true,                             // param[in]: simulated settting for user changing cpu pinning property
    {},                               // param[in]: simulated setting for current cpu_mapping_table
    {{40, 20, 0, 20, 0, 0}},          // param[in]: simulated setting for current proc_type_table
    {{1, MAIN_CORE_PROC, 20, 0, 0}},  // param[in]: simulated setting for current streams_info_table
    false,                            // param[expected out]: simulated setting for expected output
};
CpuPinningTestCase cpu_pinning_macos_mock_set_false = {
    false,
    true,
    {},
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_macos_mock_set_default = {
    true,
    false,
    {},
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_true = {
    true,
    true,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_win_mock_set_true_2 = {
    true,
    true,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_win_mock_set_false = {
    false,
    true,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_false_2 = {
    false,
    true,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_default = {
    true,
    false,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_default_2 = {
    true,
    false,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_true = {
    true,
    true,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_true_2 = {
    true,
    true,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_false = {
    false,
    true,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_false_2 = {
    false,
    true,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default = {
    false,
    false,
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
        {18, 0, 0, 18, HYPER_THREADING_PROC, 18, -1}, {19, 0, 0, 19, HYPER_THREADING_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1},       {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1},       {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 0, 0, 24, MAIN_CORE_PROC, 24, -1},       {25, 0, 0, 25, MAIN_CORE_PROC, 25, -1},
        {26, 0, 0, 26, MAIN_CORE_PROC, 26, -1},       {27, 0, 0, 27, MAIN_CORE_PROC, 27, -1},
        {28, 0, 0, 28, MAIN_CORE_PROC, 28, -1},       {29, 0, 0, 29, MAIN_CORE_PROC, 29, -1},
        {30, 0, 0, 30, MAIN_CORE_PROC, 30, -1},       {31, 0, 0, 31, MAIN_CORE_PROC, 31, -1},
        {32, 0, 0, 32, MAIN_CORE_PROC, 32, -1},       {33, 0, 0, 33, MAIN_CORE_PROC, 33, -1},
        {34, 0, 0, 34, MAIN_CORE_PROC, 34, -1},       {35, 0, 0, 35, MAIN_CORE_PROC, 35, -1},
        {36, 0, 0, 36, MAIN_CORE_PROC, 36, -1},       {37, 0, 0, 37, MAIN_CORE_PROC, 37, -1},
        {38, 0, 0, 38, MAIN_CORE_PROC, 38, -1},       {39, 0, 0, 39, MAIN_CORE_PROC, 39, -1},
    },
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default_2 = {
    true,
    false,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},         {1, 0, 0, 0, HYPER_THREADING_PROC, 1, -1},
        {2, 0, 0, 1, MAIN_CORE_PROC, 2, -1},         {3, 0, 0, 1, HYPER_THREADING_PROC, 3, -1},
        {4, 0, 0, 2, MAIN_CORE_PROC, 4, -1},         {5, 0, 0, 2, HYPER_THREADING_PROC, 5, -1},
        {6, 0, 0, 3, MAIN_CORE_PROC, 6, -1},         {7, 0, 0, 3, HYPER_THREADING_PROC, 7, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 8, -1},         {9, 0, 0, 4, HYPER_THREADING_PROC, 9, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 10, -1},       {11, 0, 0, 5, HYPER_THREADING_PROC, 11, -1},
        {12, 0, 0, 6, EFFICIENT_CORE_PROC, 12, -1},  {13, 0, 0, 7, EFFICIENT_CORE_PROC, 13, -1},
        {14, 0, 0, 8, EFFICIENT_CORE_PROC, 14, -1},  {15, 0, 0, 9, EFFICIENT_CORE_PROC, 15, -1},
        {16, 0, 0, 10, EFFICIENT_CORE_PROC, 16, -1}, {17, 0, 0, 11, EFFICIENT_CORE_PROC, 17, -1},
        {18, 0, 0, 12, EFFICIENT_CORE_PROC, 18, -1}, {19, 0, 0, 13, EFFICIENT_CORE_PROC, 19, -1},
    },
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default_3 = {
    false,
    false,
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},    {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},    {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},    {5, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},    {7, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {8, 0, 0, 8, MAIN_CORE_PROC, 8, -1},    {9, 0, 0, 9, MAIN_CORE_PROC, 9, -1},
        {10, 0, 0, 10, MAIN_CORE_PROC, 10, -1}, {11, 0, 0, 11, MAIN_CORE_PROC, 11, -1},
        {12, 0, 0, 12, MAIN_CORE_PROC, 12, -1}, {13, 0, 0, 13, MAIN_CORE_PROC, 13, -1},
        {14, 0, 0, 14, MAIN_CORE_PROC, 14, -1}, {15, 0, 0, 15, MAIN_CORE_PROC, 15, -1},
        {16, 0, 0, 16, MAIN_CORE_PROC, 16, -1}, {17, 0, 0, 17, MAIN_CORE_PROC, 17, -1},
        {18, 0, 0, 18, MAIN_CORE_PROC, 18, -1}, {19, 0, 0, 19, MAIN_CORE_PROC, 19, -1},
        {20, 0, 0, 20, MAIN_CORE_PROC, 20, -1}, {21, 0, 0, 21, MAIN_CORE_PROC, 21, -1},
        {22, 0, 0, 22, MAIN_CORE_PROC, 22, -1}, {23, 0, 0, 23, MAIN_CORE_PROC, 23, -1},
        {24, 1, 1, 24, MAIN_CORE_PROC, 24, -1}, {25, 1, 1, 25, MAIN_CORE_PROC, 25, -1},
        {26, 1, 1, 26, MAIN_CORE_PROC, 26, -1}, {27, 1, 1, 27, MAIN_CORE_PROC, 27, -1},
        {28, 1, 1, 28, MAIN_CORE_PROC, 28, -1}, {29, 1, 1, 29, MAIN_CORE_PROC, 29, -1},
        {30, 1, 1, 30, MAIN_CORE_PROC, 30, -1}, {31, 1, 1, 31, MAIN_CORE_PROC, 31, -1},
        {32, 1, 1, 32, MAIN_CORE_PROC, 32, -1}, {33, 1, 1, 33, MAIN_CORE_PROC, 33, -1},
        {34, 1, 1, 34, MAIN_CORE_PROC, 34, -1}, {35, 1, 1, 35, MAIN_CORE_PROC, 35, -1},
        {36, 1, 1, 36, MAIN_CORE_PROC, 36, -1}, {37, 1, 1, 37, MAIN_CORE_PROC, 37, -1},
        {38, 1, 1, 38, MAIN_CORE_PROC, 38, -1}, {39, 1, 1, 39, MAIN_CORE_PROC, 39, -1},
        {40, 1, 1, 40, MAIN_CORE_PROC, 40, -1}, {41, 1, 1, 41, MAIN_CORE_PROC, 41, -1},
        {42, 1, 1, 42, MAIN_CORE_PROC, 42, -1}, {43, 1, 1, 43, MAIN_CORE_PROC, 43, -1},
        {44, 1, 1, 44, MAIN_CORE_PROC, 44, -1}, {45, 1, 1, 45, MAIN_CORE_PROC, 45, -1},
        {46, 1, 1, 46, MAIN_CORE_PROC, 46, -1}, {47, 1, 1, 47, MAIN_CORE_PROC, 47, -1},
    },
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    true,
};

#if defined(__linux__)
INSTANTIATE_TEST_SUITE_P(smoke_CpuPinning,
                         CpuPinningTests,
                         ::testing::Values(cpu_pinning_linux_mock_set_true,
                                           cpu_pinning_linux_mock_set_true_2,
                                           cpu_pinning_linux_mock_set_false,
                                           cpu_pinning_linux_mock_set_false_2,
                                           cpu_pinning_linux_mock_set_default,
                                           cpu_pinning_linux_mock_set_default_2,
                                           cpu_pinning_linux_mock_set_default_3));
#elif defined(_WIN32)
INSTANTIATE_TEST_SUITE_P(smoke_CpuPinning,
                         CpuPinningTests,
                         ::testing::Values(cpu_pinning_win_mock_set_true,
                                           cpu_pinning_win_mock_set_true_2,
                                           cpu_pinning_win_mock_set_false,
                                           cpu_pinning_win_mock_set_false_2,
                                           cpu_pinning_win_mock_set_default,
                                           cpu_pinning_win_mock_set_default_2));
#else
INSTANTIATE_TEST_SUITE_P(smoke_CpuPinning,
                         CpuPinningTests,
                         ::testing::Values(cpu_pinning_macos_mock_set_true,
                                           cpu_pinning_macos_mock_set_false,
                                           cpu_pinning_macos_mock_set_default));
#endif
}  // namespace