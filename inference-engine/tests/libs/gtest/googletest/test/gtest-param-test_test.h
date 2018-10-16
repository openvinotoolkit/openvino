// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GTEST_TEST_GTEST_PARAM_TEST_TEST_H_
#define GTEST_TEST_GTEST_PARAM_TEST_TEST_H_

#include "gtest/gtest.h"

#if GTEST_HAS_PARAM_TEST

// Test fixture for testing definition and instantiation of a test
// in separate translation units.
class ExternalInstantiationTest : public ::testing::TestWithParam<int> {
};

// Test fixture for testing instantiation of a test in multiple
// translation units.
class InstantiationInMultipleTranslaionUnitsTest
    : public ::testing::TestWithParam<int> {
};

#endif  // GTEST_HAS_PARAM_TEST

#endif  // GTEST_TEST_GTEST_PARAM_TEST_TEST_H_
