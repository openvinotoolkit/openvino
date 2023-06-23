// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "preprocessing/open_cv_yuv_to_grey_tests.hpp"

using namespace ov::preprocess;

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         PreprocessingYUV2GreyTest,
                         testing::Values(CommonTestUtils::DEVICE_CPU),
                         PreprocessingYUV2GreyTest::getTestCaseName);
