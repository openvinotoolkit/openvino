// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;

namespace {

#ifdef __linux__

struct LinuxNumactlTestCase {
    std::vector<int> phy_core_list;
    std::vector<std::vector<int>> input_proc_type_table;
    std::vector<std::vector<int>> input_cpu_mapping_table;
    int _sockets;
    int _cores;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
};

class LinuxCpuMapNumactlTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxNumactlTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        int test_sockets = 0;
        int test_cores = 0;
        auto test_proc_type_table = test_data.input_proc_type_table;
        auto test_cpu_mapping_table = test_data.input_cpu_mapping_table;

        update_valid_processor_linux(test_data.phy_core_list,
                                     test_sockets,
                                     test_cores,
                                     test_proc_type_table,
                                     test_cpu_mapping_table);

        ASSERT_EQ(test_data._sockets, test_sockets);
        ASSERT_EQ(test_data._cores, test_cores);
        ASSERT_EQ(test_data._proc_type_table, test_proc_type_table);
        ASSERT_EQ(test_data._cpu_mapping_table, test_cpu_mapping_table);
    }
};

LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_1 = {
    {},  // param[in]: The logical processors selected in this simulation case does not include the physical core of
         // Pcore
    {{40, 20, 0, 20, -1, -1},
     {20, 10, 0, 10, 0, 0},
     {20, 10, 0, 10, 1, 1}},  // param[in]: The proc_type_table of simulated platform which is 2 sockets, 20 Pcores
                              // and 40 logical processors with hyper-threading enabled.
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
    },  // param[in]: This simulation case select logcial processor 0, 2, 4 and 6 which is marked as logcial core of
        // Pcore in original cpu_mapping_table.
    1,  // param[expected out]: Since all selected logical processors are in one socket, the number of sockets changes
        // to 1.
    4,  // param[expected out]: Since only 4 logical processors are selected, the number of cores changes to 4.
    {{4, 4, 0, 0, 0, 0}},  // param[expected out]: The proc_type_table changes to 4 Pcores only
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
    },  // param[expected out]: cpu_mapping_table changes to physical core of Pcore.
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_2 = {
    {1, 3, 5, 7},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {21, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {23, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {25, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {27, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
    },
    1,
    4,
    {{4, 4, 0, 0, 0, 0}},
    {
        {21, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {23, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {25, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {27, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_3 = {
    {1, 3, 5, 7},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {21, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {23, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {25, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {27, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
    },
    1,
    8,
    {{8, 8, 0, 0, 0, 0}},
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {21, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {23, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {25, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {27, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_4 = {
    {0, 2, 4, 6},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
    },
    1,
    4,
    {{8, 4, 0, 4, 0, 0}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_5 = {
    {},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {10, 1, 1, 10, HYPER_THREADING_PROC, 10, -1},
        {12, 1, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {14, 1, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {16, 1, 1, 16, HYPER_THREADING_PROC, 16, -1},
    },
    2,
    8,
    {{8, 8, 0, 0, -1, -1}, {4, 4, 0, 0, 0, 0}, {4, 4, 0, 0, 1, 1}},
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {10, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {12, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {14, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {16, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_6 = {
    {0, 2, 4, 6, 10, 12, 14, 16},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {30, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {32, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {34, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {36, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
    },
    2,
    8,
    {{8, 8, 0, 0, -1, -1}, {4, 4, 0, 0, 0, 0}, {4, 4, 0, 0, 1, 1}},
    {
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {30, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {32, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {34, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {36, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_7 = {
    {0, 2, 4, 6},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {10, 1, 1, 10, HYPER_THREADING_PROC, 10, -1},
        {12, 1, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {14, 1, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {16, 1, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
    },
    2,
    8,
    {{8, 8, 0, 0, -1, -1}, {4, 4, 0, 0, 0, 0}, {4, 4, 0, 0, 1, 1}},
    {
        {10, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {12, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {14, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {16, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
    },
};
LinuxNumactlTestCase numactl_2sockets_20cores_hyperthreading_8 = {
    {0, 2, 4, 6, 10, 12, 14, 16},
    {{40, 20, 0, 20, -1, -1}, {20, 10, 0, 10, 0, 0}, {20, 10, 0, 10, 1, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {10, 1, 1, 10, HYPER_THREADING_PROC, 10, -1},
        {12, 1, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {14, 1, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {16, 1, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {30, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {32, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {34, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {36, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
    },
    2,
    8,
    {{16, 8, 0, 8, -1, -1}, {8, 4, 0, 4, 0, 0}, {8, 4, 0, 4, 1, 1}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {10, 1, 1, 10, HYPER_THREADING_PROC, 10, -1},
        {12, 1, 1, 12, HYPER_THREADING_PROC, 12, -1},
        {14, 1, 1, 14, HYPER_THREADING_PROC, 14, -1},
        {16, 1, 1, 16, HYPER_THREADING_PROC, 16, -1},
        {20, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {22, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {24, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {26, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {30, 1, 1, 10, MAIN_CORE_PROC, 10, -1},
        {32, 1, 1, 12, MAIN_CORE_PROC, 12, -1},
        {34, 1, 1, 14, MAIN_CORE_PROC, 14, -1},
        {36, 1, 1, 16, MAIN_CORE_PROC, 16, -1},
    },
};
LinuxNumactlTestCase numactl_1sockets_16cores_hyperthreading_1 = {
    {},
    {{24, 8, 8, 8, 0, 0}},
    {
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
    1,
    4,
    {{4, 0, 4, 0, 0, 0}},
    {
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
};
LinuxNumactlTestCase numactl_1sockets_16cores_hyperthreading_2 = {
    {},
    {{24, 8, 8, 8, 0, 0}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {2, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {4, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {6, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
    1,
    8,
    {{8, 4, 4, 0, 0, 0}},
    {
        {0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {4, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {6, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
};
LinuxNumactlTestCase numactl_1sockets_16cores_hyperthreading_3 = {
    {0, 1, 2, 3},
    {{24, 8, 8, 8, 0, 0}},
    {
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
    1,
    8,
    {{8, 4, 4, 0, 0, 0}},
    {
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
};
LinuxNumactlTestCase numactl_1sockets_16cores_hyperthreading_4 = {
    {0, 1, 2, 3},
    {{24, 8, 8, 8, 0, 0}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
    1,
    8,
    {{12, 4, 4, 4, 0, 0}},
    {
        {0, 0, 0, 0, HYPER_THREADING_PROC, 0, -1},
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {2, 0, 0, 1, HYPER_THREADING_PROC, 1, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {4, 0, 0, 2, HYPER_THREADING_PROC, 2, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {6, 0, 0, 3, HYPER_THREADING_PROC, 3, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
};
LinuxNumactlTestCase numactl_1sockets_16cores_hyperthreading_5 = {
    {0, 1, 2, 3},
    {{24, 8, 8, 8, 0, 0}},
    {
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, HYPER_THREADING_PROC, 4, -1},
        {10, 0, 0, 5, HYPER_THREADING_PROC, 5, -1},
        {12, 0, 0, 6, HYPER_THREADING_PROC, 6, -1},
        {14, 0, 0, 7, HYPER_THREADING_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
    1,
    12,
    {{12, 8, 4, 0, 0, 0}},
    {
        {1, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
        {3, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
        {5, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
        {7, 0, 0, 3, MAIN_CORE_PROC, 3, -1},
        {8, 0, 0, 4, MAIN_CORE_PROC, 4, -1},
        {10, 0, 0, 5, MAIN_CORE_PROC, 5, -1},
        {12, 0, 0, 6, MAIN_CORE_PROC, 6, -1},
        {14, 0, 0, 7, MAIN_CORE_PROC, 7, -1},
        {16, 0, 0, 8, EFFICIENT_CORE_PROC, 8, -1},
        {17, 0, 0, 9, EFFICIENT_CORE_PROC, 9, -1},
        {18, 0, 0, 10, EFFICIENT_CORE_PROC, 10, -1},
        {19, 0, 0, 11, EFFICIENT_CORE_PROC, 11, -1},
    },
};

TEST_P(LinuxCpuMapNumactlTests, LinuxCpuMapNumactl) {}

INSTANTIATE_TEST_SUITE_P(CPUMap,
                         LinuxCpuMapNumactlTests,
                         testing::Values(numactl_2sockets_20cores_hyperthreading_1,
                                         numactl_2sockets_20cores_hyperthreading_2,
                                         numactl_2sockets_20cores_hyperthreading_3,
                                         numactl_2sockets_20cores_hyperthreading_4,
                                         numactl_2sockets_20cores_hyperthreading_5,
                                         numactl_2sockets_20cores_hyperthreading_6,
                                         numactl_2sockets_20cores_hyperthreading_7,
                                         numactl_2sockets_20cores_hyperthreading_8,
                                         numactl_1sockets_16cores_hyperthreading_1,
                                         numactl_1sockets_16cores_hyperthreading_2,
                                         numactl_1sockets_16cores_hyperthreading_3,
                                         numactl_1sockets_16cores_hyperthreading_4,
                                         numactl_1sockets_16cores_hyperthreading_5));

#endif
}  // namespace
