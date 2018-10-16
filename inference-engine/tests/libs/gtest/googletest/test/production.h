// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef GTEST_TEST_PRODUCTION_H_
#define GTEST_TEST_PRODUCTION_H_

#include "gtest/gtest_prod.h"

class PrivateCode {
 public:
  // Declares a friend test that does not use a fixture.
  FRIEND_TEST(PrivateCodeTest, CanAccessPrivateMembers);

  // Declares a friend test that uses a fixture.
  FRIEND_TEST(PrivateCodeFixtureTest, CanAccessPrivateMembers);

  PrivateCode();

  int x() const { return x_; }
 private:
  void set_x(int an_x) { x_ = an_x; }
  int x_;
};

#endif  // GTEST_TEST_PRODUCTION_H_
