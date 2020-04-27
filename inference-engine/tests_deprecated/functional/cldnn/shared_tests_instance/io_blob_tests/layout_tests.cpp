// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_tests.hpp"

static auto params = ::testing::Combine(
        ::testing::Values(conv_p),
        ::testing::Values(std::make_pair(Precision::FP32, 1e-5)),
        ::testing::Values(NCHW, NHWC),
        ::testing::Values(NCHW, NHWC),
        ::testing::Values(Precision::FP32, Precision::U8, Precision::I16)  // TODO: What about U16/I8/FP16?
);

PLUGING_CASE_WITH_SUFFIX(GPU, _smoke, LayoutTTTest, params);

#if defined(ENABLE_MKL_DNN)
    PLUGING_CASE(HETERO, LayoutTTTest, params);
#endif
