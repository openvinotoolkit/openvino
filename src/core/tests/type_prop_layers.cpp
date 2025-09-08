// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/ctc_greedy_decoder.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/region_yolo.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/op/roi_pooling.hpp"
using namespace std;
using namespace ov;

TEST(type_prop_layers, ctc_greedy_decoder) {
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{88, 2, 48});
    auto seq_len = make_shared<ov::op::v0::Parameter>(element::f32, Shape{88, 2});
    auto op = make_shared<op::v0::CTCGreedyDecoder>(input, seq_len, false);
    ASSERT_EQ(op->get_shape(), (Shape{2, 88, 1, 1}));
}

TEST(type_prop_layers, interpolate) {
    auto image = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 33, 65});
    auto dyn_output_shape = make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    auto output_shape = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.mode = "nearest";
    attrs.align_corners = true;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto op = make_shared<op::v0::Interpolate>(image, output_shape, attrs);
    ASSERT_EQ(op->get_shape(), (Shape{2, 2, 15, 30}));

    EXPECT_TRUE(make_shared<op::v0::Interpolate>(image, dyn_output_shape, attrs)
                    ->get_output_partial_shape(0)
                    .same_scheme(PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop_layers, region_yolo1) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 1);
    ASSERT_EQ(op->get_shape(), (Shape{1 * 125, 13, 13}));
}

TEST(type_prop_layers, region_yolo2) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 0, 0, 0, true, std::vector<int64_t>{}, 0, 2);
    ASSERT_EQ(op->get_shape(), (Shape{1 * 125 * 13, 13}));
}

TEST(type_prop_layers, region_yolo3) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 125, 13, 13});
    auto op = make_shared<op::v0::RegionYolo>(inputs, 4, 80, 1, false, std::vector<int64_t>{6, 7, 8}, 0, -1);
    ASSERT_EQ(op->get_shape(), (Shape{1, (80 + 4 + 1) * 3, 13, 13}));
}

TEST(type_prop_layers, reorg_yolo) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 24, 34, 62});
    auto op = make_shared<op::v0::ReorgYolo>(inputs, Strides{2});
    ASSERT_EQ(op->get_shape(), (Shape{2, 96, 17, 31}));
}

TEST(type_prop_layers, roi_pooling) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
    auto coords = make_shared<ov::op::v0::Parameter>(element::f32, Shape{150, 5});
    auto op = make_shared<op::v0::ROIPooling>(inputs, coords, Shape{6, 6}, 0.0625f, "max");
    ASSERT_EQ(op->get_shape(), (Shape{150, 3, 6, 6}));
}

TEST(type_prop_layers, reduce_axes_1) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 3, 4, 5});
    auto axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto op = make_shared<op::v1::ReduceProd>(inputs, axes, false);
    ASSERT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(3));
}

TEST(type_prop_layers, reduce_axes_2) {
    auto inputs = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 3, 4, 5});
    auto axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto op = make_shared<op::v1::ReduceProd>(inputs, axes, false);
    ASSERT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}
