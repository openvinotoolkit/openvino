// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_system_conf.h>

#include <common_test_utils/test_common.hpp>

#include "cpu_streams_calculation.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace ov;

namespace {

struct StreamsCalculationTestCase {
    int input_streams;
    int input_threads;
    int model_prefer_threads;
    std::vector<std::vector<int>> proc_type_table;
    std::vector<std::vector<int>> stream_info_table;
};

class StreamsCalculationTests : public CommonTestUtils::TestsCommon,
                                public testing::WithParamInterface<std::tuple<StreamsCalculationTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        std::vector<std::vector<int>> test_stream_info_table =
            ov::intel_cpu::get_streams_info_table(test_data.input_streams,
                                                  test_data.input_threads,
                                                  test_data.model_prefer_threads,
                                                  test_data.proc_type_table);

        ASSERT_EQ(test_data.stream_info_table, test_stream_info_table);
    }
};

StreamsCalculationTestCase _2sockets_104cores_latency_1 = {
    1,
    0,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{1, MAIN_CORE_PROC, 104}},
};

StreamsCalculationTestCase _2sockets_104cores_latency_2 = {
    1,
    20,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{1, MAIN_CORE_PROC, 20}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_1 = {
    0,
    0,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{26, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_2 = {
    2,
    0,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{2, MAIN_CORE_PROC, 52}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_3 = {
    0,
    20,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{5, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_4 = {
    2,
    20,
    0,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_5 = {
    0,
    0,
    1,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{104, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_6 = {
    0,
    0,
    2,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{52, MAIN_CORE_PROC, 2}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_7 = {
    0,
    0,
    8,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{13, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _2sockets_104cores_tput_8 = {
    0,
    40,
    8,
    {{208, 104, 0, 104}, {104, 52, 0, 52}, {104, 52, 0, 52}},
    {{5, MAIN_CORE_PROC, 8}},
};

StreamsCalculationTestCase _2sockets_48cores_latency_1 = {
    1,
    0,
    0,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {{1, MAIN_CORE_PROC, 48}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_1 = {
    0,
    0,
    0,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {{12, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_2 = {
    100,
    0,
    0,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {{48, MAIN_CORE_PROC, 1}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_3 = {
    0,
    100,
    0,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {{12, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _2sockets_48cores_tput_4 = {
    2,
    20,
    1,
    {{48, 48, 0, 0}, {24, 24, 0, 0}, {24, 24, 0, 0}},
    {{2, MAIN_CORE_PROC, 10}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_1 = {
    1,
    0,
    0,
    {{20, 6, 8, 6}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_2 = {
    1,
    10,
    0,
    {{20, 6, 8, 6}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_3 = {
    1,
    0,
    6,
    {{20, 6, 8, 6}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_latency_4 = {
    1,
    0,
    14,
    {{20, 6, 8, 6}},
    {{1, ALL_PROC, 14}, {0, MAIN_CORE_PROC, 6}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_1 = {
    0,
    0,
    0,
    {{20, 6, 8, 6}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_2 = {
    2,
    0,
    0,
    {{20, 6, 8, 6}},
    {{1, MAIN_CORE_PROC, 6}, {1, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_3 = {
    4,
    0,
    0,
    {{20, 6, 8, 6}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_4 = {
    0,
    12,
    0,
    {{20, 6, 8, 6}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_5 = {
    0,
    0,
    1,
    {{20, 6, 8, 6}},
    {{6, MAIN_CORE_PROC, 1}, {8, EFFICIENT_CORE_PROC, 1}, {6, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_6 = {
    0,
    0,
    2,
    {{20, 6, 8, 6}},
    {{3, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {3, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_7 = {
    100,
    0,
    0,
    {{20, 6, 8, 6}},
    {{6, MAIN_CORE_PROC, 1}, {8, EFFICIENT_CORE_PROC, 1}, {6, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_14cores_tput_8 = {
    0,
    100,
    0,
    {{20, 6, 8, 6}},
    {{2, MAIN_CORE_PROC, 3}, {2, EFFICIENT_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_1 = {
    1,
    0,
    0,
    {{12, 2, 8, 2}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_2 = {
    1,
    8,
    0,
    {{12, 2, 8, 2}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_3 = {
    1,
    0,
    2,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_latency_4 = {
    1,
    0,
    10,
    {{12, 2, 8, 2}},
    {{1, ALL_PROC, 10}, {0, MAIN_CORE_PROC, 2}, {0, EFFICIENT_CORE_PROC, 8}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_1 = {
    0,
    0,
    0,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_2 = {
    2,
    0,
    0,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}, {1, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_3 = {
    4,
    0,
    0,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}, {3, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_4 = {
    0,
    6,
    0,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_5 = {
    0,
    0,
    1,
    {{12, 2, 8, 2}},
    {{2, MAIN_CORE_PROC, 1}, {8, EFFICIENT_CORE_PROC, 1}, {2, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_10cores_tput_6 = {
    0,
    0,
    2,
    {{12, 2, 8, 2}},
    {{1, MAIN_CORE_PROC, 2}, {4, EFFICIENT_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_1 = {
    1,
    0,
    0,
    {{12, 4, 4, 4}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_2 = {
    1,
    100,
    0,
    {{12, 4, 4, 4}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_3 = {
    1,
    0,
    4,
    {{12, 4, 4, 4}},
    {{1, MAIN_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_latency_4 = {
    1,
    0,
    8,
    {{12, 4, 4, 4}},
    {{1, ALL_PROC, 8}, {0, MAIN_CORE_PROC, 4}, {0, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_1 = {
    0,
    0,
    0,
    {{12, 4, 4, 4}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}, {2, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_2 = {
    2,
    0,
    0,
    {{12, 4, 4, 4}},
    {{1, MAIN_CORE_PROC, 4}, {1, EFFICIENT_CORE_PROC, 4}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_3 = {
    4,
    0,
    0,
    {{12, 4, 4, 4}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_4 = {
    6,
    0,
    0,
    {{12, 4, 4, 4}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}, {2, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_5 = {
    0,
    6,
    0,
    {{12, 4, 4, 4}},
    {{2, MAIN_CORE_PROC, 2}, {1, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_6 = {
    0,
    8,
    0,
    {{12, 4, 4, 4}},
    {{2, MAIN_CORE_PROC, 2}, {2, EFFICIENT_CORE_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_8cores_tput_7 = {
    0,
    0,
    1,
    {{12, 4, 4, 4}},
    {{4, MAIN_CORE_PROC, 1}, {4, EFFICIENT_CORE_PROC, 1}, {4, HYPER_THREADING_PROC, 1}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_1 = {
    1,
    0,
    0,
    {{12, 6, 0, 6}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_6cores_latency_2 = {
    1,
    100,
    0,
    {{12, 6, 0, 6}},
    {{1, MAIN_CORE_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_1 = {
    0,
    0,
    0,
    {{12, 6, 0, 6}},
    {{2, MAIN_CORE_PROC, 3}, {2, HYPER_THREADING_PROC, 3}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_2 = {
    2,
    0,
    0,
    {{12, 6, 0, 6}},
    {{1, MAIN_CORE_PROC, 6}, {1, HYPER_THREADING_PROC, 6}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_3 = {
    0,
    8,
    0,
    {{12, 6, 0, 6}},
    {{3, MAIN_CORE_PROC, 2}, {1, HYPER_THREADING_PROC, 2}},
};

StreamsCalculationTestCase _1sockets_6cores_tput_4 = {
    0,
    0,
    1,
    {{12, 6, 0, 6}},
    {{6, MAIN_CORE_PROC, 1}, {6, HYPER_THREADING_PROC, 1}},
};

TEST_P(StreamsCalculationTests, StreamsCalculation) {}

INSTANTIATE_TEST_SUITE_P(StreamsInfoTable,
                         StreamsCalculationTests,
                         testing::Values(_2sockets_104cores_latency_1,
                                         _2sockets_104cores_latency_1,
                                         _2sockets_104cores_tput_1,
                                         _2sockets_104cores_tput_2,
                                         _2sockets_104cores_tput_3,
                                         _2sockets_104cores_tput_4,
                                         _2sockets_104cores_tput_5,
                                         _2sockets_104cores_tput_6,
                                         _2sockets_104cores_tput_7,
                                         _2sockets_104cores_tput_8,
                                         _2sockets_48cores_latency_1,
                                         _2sockets_48cores_tput_1,
                                         _2sockets_48cores_tput_2,
                                         _2sockets_48cores_tput_3,
                                         _2sockets_48cores_tput_4,
                                         _1sockets_14cores_latency_1,
                                         _1sockets_14cores_latency_2,
                                         _1sockets_14cores_latency_3,
                                         _1sockets_14cores_latency_4,
                                         _1sockets_14cores_tput_1,
                                         _1sockets_14cores_tput_2,
                                         _1sockets_14cores_tput_3,
                                         _1sockets_14cores_tput_4,
                                         _1sockets_14cores_tput_5,
                                         _1sockets_14cores_tput_6,
                                         _1sockets_14cores_tput_7,
                                         _1sockets_14cores_tput_8,
                                         _1sockets_10cores_latency_1,
                                         _1sockets_10cores_latency_2,
                                         _1sockets_10cores_latency_3,
                                         _1sockets_10cores_latency_4,
                                         _1sockets_10cores_tput_1,
                                         _1sockets_10cores_tput_2,
                                         _1sockets_10cores_tput_3,
                                         _1sockets_10cores_tput_4,
                                         _1sockets_10cores_tput_5,
                                         _1sockets_10cores_tput_6,
                                         _1sockets_8cores_latency_1,
                                         _1sockets_8cores_latency_2,
                                         _1sockets_8cores_latency_3,
                                         _1sockets_8cores_latency_4,
                                         _1sockets_8cores_tput_1,
                                         _1sockets_8cores_tput_2,
                                         _1sockets_8cores_tput_3,
                                         _1sockets_8cores_tput_4,
                                         _1sockets_8cores_tput_5,
                                         _1sockets_8cores_tput_6,
                                         _1sockets_8cores_tput_7,
                                         _1sockets_6cores_latency_1,
                                         _1sockets_6cores_latency_2,
                                         _1sockets_6cores_tput_1,
                                         _1sockets_6cores_tput_2,
                                         _1sockets_6cores_tput_3,
                                         _1sockets_6cores_tput_4));

}  // namespace