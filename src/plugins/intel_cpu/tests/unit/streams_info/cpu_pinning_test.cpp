// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "cpu_map_scheduling.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace testing;
using namespace ov;

namespace {

struct CpuPinningTestCase {
    bool input_cpu_pinning;
    bool input_changed;
    std::vector<std::vector<int>> input_proc_type_table;
    std::vector<std::vector<int>> input_stream_info_table;
    bool output_cpu_pinning;
};

class CpuPinningTests : public ov::test::TestsCommon,
                        public testing::WithParamInterface<std::tuple<CpuPinningTestCase>> {
public:
    void SetUp() override {
        auto test_data = std::get<0>(GetParam());

        auto test_output = ov::intel_cpu::get_cpu_pinning(test_data.input_cpu_pinning,
                                                          test_data.input_changed,
                                                          test_data.input_proc_type_table,
                                                          test_data.input_stream_info_table);

        ASSERT_EQ(test_data.output_cpu_pinning, test_data.input_cpu_pinning);
        ASSERT_EQ(test_data.output_cpu_pinning, test_output);
    }
};

TEST_P(CpuPinningTests, CpuPinning) {}

CpuPinningTestCase cpu_pinning_macos_mock_set_true = {
    true,                             // param[in]: simulated settting for cpu pinning property
    true,                             // param[in]: simulated settting for user changing cpu pinning property
    {{40, 20, 0, 20, 0, 0}},          // param[in]: simulated setting for current proc_type_table
    {{1, MAIN_CORE_PROC, 20, 0, 0}},  // param[in]: simulated setting for current streams_info_table
    false,                            // param[expected out]: simulated setting for expected output
};
CpuPinningTestCase cpu_pinning_macos_mock_set_false = {
    false,
    true,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_macos_mock_set_default = {
    true,
    false,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_true = {
    true,
    true,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_win_mock_set_true_2 = {
    true,
    true,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_false = {
    false,
    true,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_false_2 = {
    false,
    true,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_default = {
    true,
    false,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_win_mock_set_default_2 = {
    true,
    false,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_true = {
    true,
    true,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_true_2 = {
    true,
    true,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_false = {
    false,
    true,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_false_2 = {
    false,
    true,
    {{48, 48, 0, 0, -1, -1}, {24, 24, 0, 0, 0, 0}, {24, 24, 0, 0, 1, 1}},
    {{1, MAIN_CORE_PROC, 24, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default = {
    false,
    false,
    {{40, 20, 0, 20, 0, 0}},
    {{1, MAIN_CORE_PROC, 20, 0, 0}},
    true,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default_2 = {
    true,
    false,
    {{20, 6, 8, 6, 0, 0}},
    {{1, ALL_PROC, 14, 0, 0}, {0, MAIN_CORE_PROC, 6, 0, 0}, {0, EFFICIENT_CORE_PROC, 8, 0, 0}},
    false,
};
CpuPinningTestCase cpu_pinning_linux_mock_set_default_3 = {
    false,
    false,
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