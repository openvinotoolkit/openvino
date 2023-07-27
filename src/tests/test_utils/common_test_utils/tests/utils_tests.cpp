// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;

TEST(UtilsTests, get_directory_returns_root) {
    ASSERT_EQ(get_directory("/test"), "/");
}