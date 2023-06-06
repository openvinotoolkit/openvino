// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing/resize_tests.hpp"

#include <gtest/gtest.h>

using namespace ov::preprocess;

INSTANTIATE_TEST_SUITE_P(
    PreprocessingResizeTests_linear,
    PreprocessingResizeTests,
    testing::Combine(
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(ResizeAlgorithm::RESIZE_LINEAR),
        testing::Values(
            std::vector<
                float>{1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75, 4.0})),
    PreprocessingResizeTests::getTestCaseName);
