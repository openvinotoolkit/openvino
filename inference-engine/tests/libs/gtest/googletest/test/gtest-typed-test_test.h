// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GTEST_TEST_GTEST_TYPED_TEST_TEST_H_
#define GTEST_TEST_GTEST_TYPED_TEST_TEST_H_

#include "gtest/gtest.h"

#if GTEST_HAS_TYPED_TEST_P

using testing::Test;

// For testing that the same type-parameterized test case can be
// instantiated in different translation units linked together.
// ContainerTest will be instantiated in both gtest-typed-test_test.cc
// and gtest-typed-test2_test.cc.

template <typename T>
class ContainerTest : public Test {
};

TYPED_TEST_CASE_P(ContainerTest);

TYPED_TEST_P(ContainerTest, CanBeDefaultConstructed) {
  TypeParam container;
}

TYPED_TEST_P(ContainerTest, InitialSizeIsZero) {
  TypeParam container;
  EXPECT_EQ(0U, container.size());
}

REGISTER_TYPED_TEST_CASE_P(ContainerTest,
                           CanBeDefaultConstructed, InitialSizeIsZero);

#endif  // GTEST_HAS_TYPED_TEST_P

#endif  // GTEST_TEST_GTEST_TYPED_TEST_TEST_H_
