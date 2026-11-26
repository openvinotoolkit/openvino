// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "micro_utils.hpp"

namespace {

struct grf_mode_test_params {
    int grf_min;
    cldnn::gpu_arch arch;
    const char* expected_option;
};

class grf_mode_build_option_test : public testing::TestWithParam<grf_mode_test_params> {};

TEST_P(grf_mode_build_option_test, selects_mode_for_architecture_and_grf_requirement) {
    const auto& param = GetParam();
    EXPECT_EQ(micro::get_grf_mode_build_option(param.grf_min, param.arch), param.expected_option);
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    grf_mode_build_option_test,
    testing::Values(
        grf_mode_test_params{128, cldnn::gpu_arch::xe3p, ""},
        grf_mode_test_params{129, cldnn::gpu_arch::xe3p, " -cl-intel-256-GRF-per-thread"},
        grf_mode_test_params{256, cldnn::gpu_arch::xe3p, " -cl-intel-256-GRF-per-thread"},
        grf_mode_test_params{257, cldnn::gpu_arch::xe3p, " -cl-intel-512-GRF-per-thread"},
        grf_mode_test_params{512, cldnn::gpu_arch::xe3p, " -cl-intel-512-GRF-per-thread"},
        grf_mode_test_params{257, cldnn::gpu_arch::xe3, " -cl-intel-256-GRF-per-thread"},
        grf_mode_test_params{512, cldnn::gpu_arch::xe3, " -cl-intel-256-GRF-per-thread"}));

}  // namespace
