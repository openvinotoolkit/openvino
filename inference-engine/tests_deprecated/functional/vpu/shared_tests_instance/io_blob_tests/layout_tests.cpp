// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_tests.hpp"
#include "vpu_tests_config.hpp"

static auto params_myriad = ::testing::Combine(
        ::testing::Values(conv_p),
        ::testing::Values(std::make_pair(Precision::FP16, 1e-1)),
        ::testing::Values(NCHW, NHWC),
        ::testing::Values(NCHW, NHWC),
        ::testing::Values(Precision::FP32, Precision::U8)  // TODO: What about U16/I8/FP16?
);

// TODO: rewrite to ngraph to have reshape functionality
// VPU_PLUGING_CASE_WITH_SUFFIX(_nightly, LayoutTTTest, params_myriad);
