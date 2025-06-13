// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/log_dispatch.hpp"

#include <gtest/gtest.h>

namespace ov::test {
TEST(log_dispatch, output_to_buffer) {
    auto& log_out = ov::util::LogDispatch::Out();
    std::cout << "std::cout\n";
    log_out << "ov::util::log_dispatch out" << std::endl;
}

TEST(log_dispatch, error_to_buffer) {
    auto& log_err = ov::util::LogDispatch::Err();
    std::cerr << "std::cerr\n";
    log_err << "ov::util::log_dispatch err" << std::endl;
}

}  // namespace ov::test
