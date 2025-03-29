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

#ifdef __APPLE__

struct MacOSCpuMapTestCase {
    int _processors;
    int _numa_nodes;
    int _sockets;
    int _cores;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::pair<std::string, uint64_t>> system_info_table;
};

class MacOSCpuMapParserTests : public ov::test::TestsCommon,
                               public testing::WithParamInterface<std::tuple<MacOSCpuMapTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        int test_processors = 0;
        int test_numa_nodes = 0;
        int test_sockets = 0;
        int test_cores = 0;
        std::vector<std::vector<int>> test_proc_type_table;

        ov::parse_processor_info_macos(test_data.system_info_table,
                                       test_processors,
                                       test_numa_nodes,
                                       test_sockets,
                                       test_cores,
                                       test_proc_type_table);

        ASSERT_EQ(test_data._processors, test_processors);
        ASSERT_EQ(test_data._numa_nodes, test_numa_nodes);
        ASSERT_EQ(test_data._sockets, test_sockets);
        ASSERT_EQ(test_data._cores, test_cores);
        ASSERT_EQ(test_data._proc_type_table, test_proc_type_table);
    }
};

MacOSCpuMapTestCase test_case_arm_1 = {
    8,                     // param[expected out]: total 8 logcial processors on this simulated platform
    1,                     // param[expected out]: total 1 numa nodes on this simulated platform
    1,                     // param[expected out]: total 1 sockets on this simulated platform
    8,                     // param[expected out]: total 8 CPU cores on this simulated platform
    {{8, 4, 4, 0, 0, 0}},  // param[expected out]: The proc_type_table of this simulated platform
    {
        {"hw.ncpu", 8},
        {"hw.physicalcpu", 8},
        {"hw.optional.arm64", 1},
        {"hw.perflevel0.physicalcpu", 4},
        {"hw.perflevel1.physicalcpu", 4},
    },  // param[in]: The system information table of this simulated platform
};

MacOSCpuMapTestCase test_case_arm_2 = {
    8,
    1,
    1,
    8,
    {{8, 4, 4, 0, 0, 0}},
    {
        {"hw.ncpu", 8},
        {"hw.physicalcpu", 8},
        {"hw.optional.arm64", 1},
    },
};

MacOSCpuMapTestCase test_case_arm_3 = {
    8,
    1,
    1,
    8,
    {{8, 4, 4, 0, 0, 0}},
    {
        {"hw.ncpu", 8},
        {"hw.optional.arm64", 1},
    },
};

MacOSCpuMapTestCase test_case_x86_1 = {
    12,
    1,
    1,
    6,
    {{12, 6, 0, 6, 0, 0}},
    {{"hw.ncpu", 12}, {"hw.physicalcpu", 6}},
};

MacOSCpuMapTestCase test_case_x86_2 = {
    12,
    1,
    1,
    12,
    {{12, 12, 0, 0, 0, 0}},
    {{"hw.ncpu", 12}},
};

TEST_P(MacOSCpuMapParserTests, MacOS) {}

INSTANTIATE_TEST_SUITE_P(
    CPUMap,
    MacOSCpuMapParserTests,
    testing::Values(test_case_arm_1, test_case_arm_2, test_case_arm_3, test_case_x86_1, test_case_x86_2));

#endif
}  // namespace
