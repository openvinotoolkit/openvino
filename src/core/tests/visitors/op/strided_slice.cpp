// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/strided_slice.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, strided_slice_op) {
    NodeBuilder::opset().insert<ov::op::v1::StridedSlice>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto begin = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto end = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto stride = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    auto begin_mask = std::vector<int64_t>{0, 0};
    auto end_mask = std::vector<int64_t>{0, 0};
    auto new_axis_mask = std::vector<int64_t>{0, 0};
    auto shrink_axis_mask = std::vector<int64_t>{0, 0};
    auto ellipsis_mask = std::vector<int64_t>{0, 0};

    auto strided_slice = make_shared<ov::op::v1::StridedSlice>(data,
                                                               begin,
                                                               end,
                                                               stride,
                                                               begin_mask,
                                                               end_mask,
                                                               new_axis_mask,
                                                               shrink_axis_mask,
                                                               ellipsis_mask);
    NodeBuilder builder(strided_slice, {data, begin, end, stride});
    auto g_strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(builder.create());

    EXPECT_EQ(g_strided_slice->get_begin_mask(), strided_slice->get_begin_mask());
    EXPECT_EQ(g_strided_slice->get_end_mask(), strided_slice->get_end_mask());
    EXPECT_EQ(g_strided_slice->get_new_axis_mask(), strided_slice->get_new_axis_mask());
    EXPECT_EQ(g_strided_slice->get_shrink_axis_mask(), strided_slice->get_shrink_axis_mask());
    EXPECT_EQ(g_strided_slice->get_ellipsis_mask(), strided_slice->get_ellipsis_mask());
}
