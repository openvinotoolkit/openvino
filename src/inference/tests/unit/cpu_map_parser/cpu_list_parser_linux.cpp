// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;

namespace {

#ifdef __linux__

using CpuRanges = std::vector<std::pair<int, int>>;

struct LinuxCpuListTestCase {
    std::string cpu_list;
    CpuRanges cpu_ranges;
};

class LinuxCpuListParserTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<LinuxCpuListTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        CpuRanges test_cpu_ranges;

        ASSERT_TRUE(ov::parse_cpu_list_linux(test_data.cpu_list, test_cpu_ranges));
        ASSERT_EQ(test_data.cpu_ranges, test_cpu_ranges);
    }
};

class LinuxCpuListRejectTests : public ov::test::TestsCommon,
                                public testing::WithParamInterface<std::tuple<std::string>> {
public:
    void SetUp() override {
        const auto& cpu_list = std::get<0>(GetParam());

        CpuRanges test_cpu_ranges = {{7, 7}};

        ASSERT_FALSE(ov::parse_cpu_list_linux(cpu_list, test_cpu_ranges));
        ASSERT_TRUE(test_cpu_ranges.empty());
    }
};

TEST_P(LinuxCpuListParserTests, LinuxCpuList) {}
TEST_P(LinuxCpuListRejectTests, LinuxCpuList) {}

LinuxCpuListTestCase cpu_list_empty = {"", {}};
LinuxCpuListTestCase cpu_list_single = {"3", {{3, 3}}};
LinuxCpuListTestCase cpu_list_range = {"0-3", {{0, 3}}};
LinuxCpuListTestCase cpu_list_singles = {"0,2", {{0, 0}, {2, 2}}};
LinuxCpuListTestCase cpu_list_range_single = {"0-1,3", {{0, 1}, {3, 3}}};
LinuxCpuListTestCase cpu_list_single_range = {"0,2-3", {{0, 0}, {2, 3}}};
LinuxCpuListTestCase cpu_list_ranges = {"0-1,3-4", {{0, 1}, {3, 4}}};
LinuxCpuListTestCase cpu_list_sparse = {"2,5,10,15", {{2, 2}, {5, 5}, {10, 10}, {15, 15}}};
LinuxCpuListTestCase cpu_list_two_sockets = {"0-31,64-95", {{0, 31}, {64, 95}}};
LinuxCpuListTestCase cpu_list_no_expansion = {"0-2147483647", {{0, std::numeric_limits<int>::max()}}};

INSTANTIATE_TEST_SUITE_P(CPUMap,
                         LinuxCpuListParserTests,
                         testing::Values(cpu_list_empty,
                                         cpu_list_single,
                                         cpu_list_range,
                                         cpu_list_singles,
                                         cpu_list_range_single,
                                         cpu_list_single_range,
                                         cpu_list_ranges,
                                         cpu_list_sparse,
                                         cpu_list_two_sockets,
                                         cpu_list_no_expansion));

INSTANTIATE_TEST_SUITE_P(CPUMap,
                         LinuxCpuListRejectTests,
                         testing::Values(std::string(","),
                                         std::string("0,"),
                                         std::string(",0"),
                                         std::string("0,,1"),
                                         std::string("-1"),
                                         std::string("1-"),
                                         std::string("2-1"),
                                         std::string("1-2-3"),
                                         std::string("0x1"),
                                         std::string("1 2"),
                                         std::string("1:2"),
                                         std::string("2147483648"),
                                         std::string("0-2147483648")));

struct LinuxNodeInfoTestCase {
    std::vector<std::string> node_info_table;
    std::vector<std::vector<int>> cpu_mapping_table;
    int numa_nodes;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<int> numa_node_ids;
};

class LinuxCpuMapNodeInfoTests : public ov::test::TestsCommon,
                                 public testing::WithParamInterface<std::tuple<LinuxNodeInfoTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        int test_numa_nodes = 0;
        int test_sockets = static_cast<int>(test_data.node_info_table.size());
        std::vector<std::vector<int>> test_proc_type_table;
        std::vector<std::vector<int>> test_cpu_mapping_table = test_data.cpu_mapping_table;

        ov::parse_node_info_linux(test_data.node_info_table,
                                  test_numa_nodes,
                                  test_sockets,
                                  test_proc_type_table,
                                  test_cpu_mapping_table);

        ASSERT_EQ(test_data.numa_nodes, test_numa_nodes);
        ASSERT_EQ(test_data.proc_type_table, test_proc_type_table);
        ASSERT_EQ(test_data.numa_node_ids.size(), test_cpu_mapping_table.size());
        for (size_t i = 0; i < test_cpu_mapping_table.size(); i++) {
            ASSERT_EQ(test_data.cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID],
                      test_cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
            ASSERT_EQ(test_data.numa_node_ids[i], test_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]);
        }
    }
};

TEST_P(LinuxCpuMapNodeInfoTests, LinuxCpuMap) {}

LinuxNodeInfoTestCase node_info_single = {
    {"3"},
    {{3, 0, 0, 3, MAIN_CORE_PROC, 3, -1}},
    1,
    {{1, 1, 0, 0, 0, 0, 0}},
    {0},
};
LinuxNodeInfoTestCase node_info_single_range = {
    {"0,2-3", "1"},
    {{0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
     {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
     {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
     {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1}},
    2,
    {{4, 4, 0, 0, 0, -1, 0}, {3, 3, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1, 0}},
    {0, 1, 0, 0},
};
LinuxNodeInfoTestCase node_info_range_single = {
    {"0-1,3", "2"},
    {{0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
     {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
     {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
     {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1}},
    2,
    {{4, 4, 0, 0, 0, -1, 0}, {3, 3, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 1, 0}},
    {0, 0, 1, 0},
};
LinuxNodeInfoTestCase node_info_interleaved = {
    {"0,2", "1,3"},
    {{0, 0, 0, 0, MAIN_CORE_PROC, 0, -1},
     {1, 0, 0, 1, MAIN_CORE_PROC, 1, -1},
     {2, 0, 0, 2, MAIN_CORE_PROC, 2, -1},
     {3, 0, 0, 3, MAIN_CORE_PROC, 3, -1}},
    2,
    {{4, 4, 0, 0, 0, -1, 0}, {2, 2, 0, 0, 0, 0, 0}, {2, 2, 0, 0, 0, 1, 0}},
    {0, 1, 0, 1},
};

INSTANTIATE_TEST_SUITE_P(
    CPUMap,
    LinuxCpuMapNodeInfoTests,
    testing::Values(node_info_single, node_info_single_range, node_info_range_single, node_info_interleaved));

#endif
}  // namespace
