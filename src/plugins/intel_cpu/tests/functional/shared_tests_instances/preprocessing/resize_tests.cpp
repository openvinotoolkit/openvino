// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing/resize_tests.hpp"

#include <gtest/gtest.h>

using namespace ov::preprocess;

INSTANTIATE_TEST_SUITE_P(smoke_Preprocessing,
                         PreprocessingResizeTests,
                         testing::Values(ov::test::utils::DEVICE_CPU),
                         PreprocessingResizeTests::getTestCaseName);
