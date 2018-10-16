// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

// Google Mock - a framework for writing C++ mock classes.
//
// This file implements some matchers that depend on gmock-generated-matchers.h.
//
// Note that tests are implemented in gmock-matchers_test.cc rather than
// gmock-more-matchers-test.cc.

#ifndef GMOCK_GMOCK_MORE_MATCHERS_H_
#define GMOCK_GMOCK_MORE_MATCHERS_H_

#include "gmock/gmock-generated-matchers.h"

namespace testing {

// Defines a matcher that matches an empty container. The container must
// support both size() and empty(), which all STL-like containers provide.
MATCHER(IsEmpty, negation ? "isn't empty" : "is empty") {
  if (arg.empty()) {
    return true;
  }
  *result_listener << "whose size is " << arg.size();
  return false;
}

}  // namespace testing

#endif  // GMOCK_GMOCK_MORE_MATCHERS_H_
