// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <openvino/reference/xattention.hpp>

double DEFAULT_THRESHOLD = 0.8;
size_t DEFAULT_BLOCK_SIZE = 32;
size_t DEFAULT_STRIDE = 8;

TEST(XAttention, SelectsBlocksWithoutThrowing) {
    double threshold = 0.8;
    size_t block_size = 32;
    size_t stride = 8;
    ov::reference::XAttentionBlockSelector<double> selector(threshold, block_size, stride);
    ov::Shape q_shape = {4, 64, 128};
    ov::Shape k_shape = {4, 64, 128};
    auto q_buf = selector.allocate_buf(q_shape);
    auto k_buf = selector.allocate_buf(q_shape);

    EXPECT_NO_THROW(selector.select_blocks(q_buf.get(), q_shape, k_buf.get(), k_shape));
};

struct DiagonalReshapeTestData {
    std::vector<double> in_data;
    std::vector<double> in_shape;
    bool is_antidiagonal;
    size_t stride;
    std::vector<double> ref_out_data;
    std::vector<double> out_shape;
};

TEST_P(XAttention, ReshapesDiagonally) {
    
}
