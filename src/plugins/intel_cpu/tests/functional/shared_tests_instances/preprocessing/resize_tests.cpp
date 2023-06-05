// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing/resize_tests.hpp"

#include <gtest/gtest.h>

using namespace ov::preprocess;

const std::vector<ResizeAlgorithm> algos = {ResizeAlgorithm::RESIZE_LINEAR};

INSTANTIATE_TEST_SUITE_P(PreprocessingResizeTests,
                         PreprocessingResizeTests,
                         testing::Combine(testing::Values(CommonTestUtils::DEVICE_CPU),
                                          testing::ValuesIn(algos)),
                         PreprocessingResizeTests::getTestCaseName);
