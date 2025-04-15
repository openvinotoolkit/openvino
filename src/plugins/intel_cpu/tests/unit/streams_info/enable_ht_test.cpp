// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_map_scheduling.hpp"
#include "cpu_streams_calculation.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace testing;
using namespace ov;

namespace {

struct UseHTTestCase {
    bool input_ht_value;
    bool input_ht_changed;
    std::string input_pm_hint;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> result_table;
    bool output_ht_value;
};

class UseHTTests : public ov::test::TestsCommon, public testing::WithParamInterface<std::tuple<UseHTTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_result_table =
            ov::intel_cpu::apply_hyper_threading(test_data.input_ht_value,
                                                 test_data.input_ht_changed,
                                                 test_data.input_pm_hint,
                                                 test_data.proc_type_table);

        ASSERT_EQ(test_data.result_table, test_result_table);
        ASSERT_EQ(test_data.input_ht_value, test_data.output_ht_value);
    }
};

UseHTTestCase _2sockets_false_latency = {
    false,
    true,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_true_latency = {
    true,
    true,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    true,
};

UseHTTestCase _2sockets_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    true,
};

UseHTTestCase _2sockets_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _2sockets_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, 104, 0, 0}, {52, 52, 0, 0}, {52, 52, 0, 0}},
    false,
};

UseHTTestCase _1sockets_1_false_latency = {
    false,
    true,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_true_latency = {
    true,
    true,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_1_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{20, 6, 8, 6}},
    {{14, 6, 8, 0}},
    false,
};

UseHTTestCase _1sockets_1_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{20, 6, 8, 6}},
    {{20, 6, 8, 6}},
    true,
};

UseHTTestCase _1sockets_2_false_latency = {
    false,
    true,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_false_throughput = {
    false,
    true,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_true_latency = {
    true,
    true,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_true_throughput = {
    true,
    true,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_default_1_latency = {
    false,
    false,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_default_1_throughput = {
    false,
    false,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

UseHTTestCase _1sockets_2_default_2_latency = {
    true,
    false,
    "LATENCY",
    {{12, 6, 0, 6}},
    {{6, 6, 0, 0}},
    false,
};

UseHTTestCase _1sockets_2_default_2_throughput = {
    true,
    false,
    "THROUGHPUT",
    {{12, 6, 0, 6}},
    {{12, 6, 0, 6}},
    true,
};

TEST_P(UseHTTests, UseHT) {}

INSTANTIATE_TEST_SUITE_P(UseHTTable,
                         UseHTTests,
                         testing::Values(_2sockets_false_latency,
                                         _2sockets_true_latency,
                                         _2sockets_default_1_latency,
                                         _2sockets_default_2_latency,
                                         _1sockets_1_false_latency,
                                         _1sockets_1_true_latency,
                                         _1sockets_1_default_1_latency,
                                         _1sockets_1_default_2_latency,
                                         _1sockets_2_false_latency,
                                         _1sockets_2_true_latency,
                                         _1sockets_2_default_1_latency,
                                         _1sockets_2_default_2_latency,
                                         _2sockets_false_throughput,
                                         _2sockets_true_throughput,
                                         _2sockets_default_1_throughput,
                                         _2sockets_default_2_throughput,
                                         _1sockets_1_false_throughput,
                                         _1sockets_1_true_throughput,
                                         _1sockets_1_default_1_throughput,
                                         _1sockets_1_default_2_throughput,
                                         _1sockets_2_false_throughput,
                                         _1sockets_2_true_throughput,
                                         _1sockets_2_default_1_throughput,
                                         _1sockets_2_default_2_throughput));

}  // namespace