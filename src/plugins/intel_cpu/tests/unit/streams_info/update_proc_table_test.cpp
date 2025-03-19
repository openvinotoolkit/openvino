// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_streams_calculation.hpp"

using namespace testing;

namespace ov {

namespace intel_cpu {

struct LinuxSortProcTableTestCase {
    int current_numa_node_id;
    std::vector<std::vector<int>> _proc_type_table_input;
    std::vector<std::vector<int>> _proc_type_table_output;
};

class LinuxSortProcTableTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxSortProcTableTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_proc_type_table = test_data._proc_type_table_input;

        sort_table_by_numa_node_id(test_data.current_numa_node_id, test_proc_type_table);

        ASSERT_EQ(test_proc_type_table, test_data._proc_type_table_output);
    }
};

LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_1 = {
    0,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_2 = {
    1,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_3 = {
    2,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}},
};
LinuxSortProcTableTestCase proc_table_2sockets_24cores_hyperthreading_4 = {
    3,
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}, {12, 6, 0, 6, 3, 1}},
    {{48, 24, 0, 24, -1, -1}, {12, 6, 0, 6, 3, 1}, {12, 6, 0, 6, 0, 0}, {12, 6, 0, 6, 1, 0}, {12, 6, 0, 6, 2, 1}},
};
LinuxSortProcTableTestCase proc_table_1sockets_mock = {
    3,
    {{48, 24, 0, 24, 0, 0}},
    {{48, 24, 0, 24, 0, 0}},
};

TEST_P(LinuxSortProcTableTests, UpdateProcTable) {}

INSTANTIATE_TEST_SUITE_P(UpdateProcTableList,
                         LinuxSortProcTableTests,
                         testing::Values(proc_table_2sockets_24cores_hyperthreading_1,
                                         proc_table_2sockets_24cores_hyperthreading_2,
                                         proc_table_2sockets_24cores_hyperthreading_3,
                                         proc_table_2sockets_24cores_hyperthreading_4,
                                         proc_table_1sockets_mock));
}  // namespace intel_cpu
}  // namespace ov
