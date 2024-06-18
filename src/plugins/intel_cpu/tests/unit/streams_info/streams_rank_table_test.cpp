// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_streams_calculation.hpp"

using namespace testing;
using namespace ov;

namespace {

struct StreamsRankTestCase {
    std::vector<std::vector<int>> stream_info_table;
    int rank_level;
    int num_sub_streams;
    std::vector<std::vector<int>> stream_rank_table;
};

class StreamsRankTests : public ov::test::TestsCommon,
                         public testing::WithParamInterface<std::tuple<StreamsRankTestCase>> {
public:
    void SetUp() override {
        const auto& test_data = std::get<0>(GetParam());

        int test_num_sub_streams;
        std::vector<std::vector<int>> test_stream_rank_table =
            ov::intel_cpu::get_streams_rank_table(test_data.stream_info_table,
                                                  test_data.rank_level,
                                                  test_num_sub_streams);

        ASSERT_EQ(test_data.stream_rank_table, test_stream_rank_table);
        ASSERT_EQ(test_data.num_sub_streams, test_num_sub_streams);
    }
};

StreamsRankTestCase _2sockets_mock_1 = {
    {{1, ALL_PROC, 208, -1, -1},
     {-1, ALL_PROC, 104, 0, 0},
     {0, MAIN_CORE_PROC, 52, 0, 0},
     {0, HYPER_THREADING_PROC, 52, 0, 0},
     {-1, ALL_PROC, 104, 1, 1},
     {0, MAIN_CORE_PROC, 52, 1, 1},
     {0, HYPER_THREADING_PROC, 52, 1, 1}},  // param[in]: the expected result of streams_info_table in this simulation
    1,                                      // param[in]: the number of rank level in this simulation
    2,                                      // param[expected out]: the number of sub stream in this simulation
    {{0}, {1}},  // param[expected out]: the expected result of streams_rank_table in thissimulation
};
StreamsRankTestCase _2sockets_mock_2 = {
    {{1, MAIN_CORE_PROC, 104, -1, -1}, {-1, MAIN_CORE_PROC, 52, 0, 0}, {-1, MAIN_CORE_PROC, 52, 1, 1}},
    1,
    2,
    {{0}, {1}},
};
StreamsRankTestCase _2sockets_mock_3 = {
    {{1, ALL_PROC, 208, -1, -1},
     {-1, ALL_PROC, 104, -1, 0},
     {0, MAIN_CORE_PROC, 26, 0, 0},
     {0, MAIN_CORE_PROC, 26, 1, 0},
     {0, HYPER_THREADING_PROC, 26, 0, 0},
     {0, HYPER_THREADING_PROC, 26, 1, 0},
     {-1, ALL_PROC, 104, -1, 1},
     {0, MAIN_CORE_PROC, 26, 2, 1},
     {0, MAIN_CORE_PROC, 26, 3, 1},
     {0, HYPER_THREADING_PROC, 26, 2, 1},
     {0, HYPER_THREADING_PROC, 26, 3, 1}},
    1,
    2,
    {{0}, {1}},
};
StreamsRankTestCase _2sockets_mock_4 = {
    {{1, MAIN_CORE_PROC, 104, -1, -1},
     {-1, MAIN_CORE_PROC, 26, 0, 0},
     {-1, MAIN_CORE_PROC, 26, 1, 0},
     {-1, MAIN_CORE_PROC, 26, 2, 1},
     {-1, MAIN_CORE_PROC, 26, 3, 1}},
    1,
    4,
    {{0}, {1}, {2}, {3}},
};
StreamsRankTestCase _2sockets_mock_5 = {
    {{1, MAIN_CORE_PROC, 104, -1, -1},
     {-1, MAIN_CORE_PROC, 26, 0, 0},
     {-1, MAIN_CORE_PROC, 26, 1, 0},
     {-1, MAIN_CORE_PROC, 26, 2, 1},
     {-1, MAIN_CORE_PROC, 26, 3, 1}},
    2,
    4,
    {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
};
StreamsRankTestCase _1sockets_mock_1 = {
    {{1, MAIN_CORE_PROC, 16, 0, 0}, {-4, MAIN_CORE_PROC, 4, 0, 0}},
    1,
    4,
    {{0}, {1}, {2}, {3}},
};
StreamsRankTestCase _1sockets_mock_2 = {
    {{1, MAIN_CORE_PROC, 16, 0, 0}, {-4, MAIN_CORE_PROC, 4, 0, 0}},
    2,
    4,
    {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
};
StreamsRankTestCase _1sockets_mock_3 = {
    {{1, MAIN_CORE_PROC, 32, 0, 0}, {-8, MAIN_CORE_PROC, 4, 0, 0}},
    2,
    8,
    {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 0}, {1, 1}, {1, 2}, {1, 3}},
};

TEST_P(StreamsRankTests, StreamsRankTest) {}

INSTANTIATE_TEST_SUITE_P(StreamsRankTable,
                         StreamsRankTests,
                         testing::Values(_2sockets_mock_1,
                                         _2sockets_mock_2,
                                         _2sockets_mock_3,
                                         _2sockets_mock_4,
                                         _2sockets_mock_5,
                                         _1sockets_mock_1,
                                         _1sockets_mock_2,
                                         _1sockets_mock_3));

}  // namespace
