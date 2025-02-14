// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "lib_close.hpp"

#include <gtest/gtest.h>

#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;

INSTANTIATE_TEST_SUITE_P(
    Paddle,
    FrontendLibCloseTest,
    Values(std::make_tuple("paddle",
                           path_join({TEST_PADDLE_MODELS_DIRNAME, "conv2d_relu/conv2d_relu.pdmodel"}).string(),
                           "conv2d_0.tmp_0")),
    FrontendLibCloseTest::get_test_case_name);
