// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/test_common.hpp>

#include "openvino/runtime/threading/istreams_executor.hpp"
#include "os/cpu_map_info.hpp"

using namespace testing;
using namespace ov;
using namespace threading;

namespace {

#if defined(__linux__) || defined(_WIN32)

struct MakeDefaultMultiThreadsTestCase {
    std::vector<std::vector<int>> _proc_type_table;
    int _num_streams;
    std::vector<std::vector<int>> _streams_info_table;
};

class MakeDefaultMultiThreadsTest : public ov::test::TestsCommon,
                                    public testing::WithParamInterface<std::tuple<MakeDefaultMultiThreadsTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        CPU& cpu = cpu_info();
        cpu._org_proc_type_table = test_data._proc_type_table;
        cpu._proc_type_table = test_data._proc_type_table;
        cpu._numa_nodes =
            test_data._proc_type_table.size() > 1 ? static_cast<int>(test_data._proc_type_table.size()) - 1 : 1;

        ov::threading::IStreamsExecutor::Config config{"make default multi threads test", test_data._num_streams};
        auto streamsConfig = ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(config);

        ASSERT_EQ(streamsConfig.get_streams_info_table(), test_data._streams_info_table);
    }
};

MakeDefaultMultiThreadsTestCase _1sockets_streams_1 = {
    // param[in]: proc_type_table, {total processors, number of physical processors, number of Efficient processors,
    // number of hyper threading processors}
    {
        {12, 6, 0, 6, 0, 0},
    },
    1,  // param[in]: the number of streams
    // param[out]: streams info table
    {
        {1, 0, 12, 0, 0},
        {0, 1, 6, 0, 0},
        {0, 3, 6, 0, 0},
    },
};

MakeDefaultMultiThreadsTestCase _1sockets_streams_2 = {
    {
        {12, 6, 0, 6, 0, 0},
    },
    2,
    {
        {1, 1, 6, 0, 0},
        {1, 3, 6, 0, 0},
    },
};

MakeDefaultMultiThreadsTestCase _2sockets_streams_1 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
    1,
    {
        {1, 0, 72, -1, -1},
        {0, 1, 18, 0, 0},
        {0, 1, 18, 1, 1},
        {0, 3, 18, 0, 0},
        {0, 3, 18, 1, 1},
    },
};

MakeDefaultMultiThreadsTestCase _2sockets_streams_4 = {
    {
        {72, 36, 0, 36, -1, -1},
        {36, 18, 0, 18, 0, 0},
        {36, 18, 0, 18, 1, 1},
    },
    4,
    {
        {1, 1, 18, 0, 0},
        {1, 1, 18, 1, 1},
        {1, 3, 18, 0, 0},
        {1, 3, 18, 1, 1},
    },
};

MakeDefaultMultiThreadsTestCase _pecore24_streams_1 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    1,
    {
        {1, 0, 24, 0, 0},
        {0, 1, 8, 0, 0},
        {0, 2, 8, 0, 0},
        {0, 3, 8, 0, 0},
    },
};

MakeDefaultMultiThreadsTestCase _pecore24_streams_3 = {
    {
        {24, 8, 8, 8, 0, 0},
    },
    3,
    {
        {1, 1, 8, 0, 0},
        {1, 2, 8, 0, 0},
        {1, 3, 8, 0, 0},
    },
};

MakeDefaultMultiThreadsTestCase _pecore32_streams_1 = {
    {
        {32, 8, 16, 8, 0, 0},
    },
    1,
    {
        {1, 0, 32, 0, 0},
        {0, 1, 8, 0, 0},
        {0, 2, 16, 0, 0},
        {0, 3, 8, 0, 0},
    },
};

MakeDefaultMultiThreadsTestCase _pecore32_streams_5 = {
    {
        {32, 8, 16, 8, 0, 0},
    },
    5,
    {
        {1, 1, 5, 0, 0},
        {3, 2, 5, 0, 0},
        {1, 3, 5, 0, 0},
    },
};

TEST_P(MakeDefaultMultiThreadsTest, MakeDefaultMultiThreads) {}

INSTANTIATE_TEST_SUITE_P(smoke_MakeDefaultMultiThreads,
                         MakeDefaultMultiThreadsTest,
                         testing::Values(_1sockets_streams_1,
                                         _1sockets_streams_2,
                                         _2sockets_streams_1,
                                         _2sockets_streams_4,
                                         _pecore24_streams_1,
                                         _pecore24_streams_3,
                                         _pecore32_streams_1,
                                         _pecore32_streams_5));
#endif
}  // namespace
