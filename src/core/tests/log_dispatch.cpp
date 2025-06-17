// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/log_dispatch.hpp"

#include <gtest/gtest.h>

namespace ov::test {
TEST(log_dispatch, ov_cout_to_console) {
    std::cout.flush();
    const auto cout_buf = std::cout.rdbuf();
    std::stringstream sstr;
    std::cout.rdbuf(sstr.rdbuf());
    {
        ov_cout << "TEST 123" << std::endl;
        EXPECT_EQ(sstr.str(), "TEST 123\n");

        std::cout << "test abc" << std::endl;
        EXPECT_EQ(sstr.str(), "TEST 123\ntest abc\n");
    }
    std::cout.rdbuf(cout_buf);
}
}  // namespace ov::test
