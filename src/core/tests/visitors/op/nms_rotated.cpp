// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/nms_rotated.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, nms_rotated_v13_default_attributes) {
    NodeBuilder::get_ops().register_factory<ov::op::v13::NMSRotated>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 5});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});
    auto max_out = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto iou_tresh = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto score_tresh = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    auto nms = make_shared<ov::op::v13::NMSRotated>(boxes, scores, max_out, iou_tresh, score_tresh);

    NodeBuilder builder(nms, {boxes, scores, max_out, iou_tresh, score_tresh});
    auto g_nms = ov::as_type_ptr<ov::op::v13::NMSRotated>(builder.create());

    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type_attr(), nms->get_output_type_attr());
    EXPECT_EQ(g_nms->get_clockwise(), nms->get_clockwise());
}

TEST(attributes, nms_rotated_v13_custom_attributes) {
    NodeBuilder::get_ops().register_factory<ov::op::v13::NMSRotated>();
    auto boxes = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 5});
    auto scores = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1, 1});
    auto max_out = make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto iou_tresh = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto score_tresh = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    auto sort_results_desc = false;
    auto output_elem_type = element::i32;
    auto clockwise = false;
    auto nms = make_shared<ov::op::v13::NMSRotated>(boxes,
                                                    scores,
                                                    max_out,
                                                    iou_tresh,
                                                    score_tresh,
                                                    sort_results_desc,
                                                    output_elem_type,
                                                    clockwise);

    NodeBuilder builder(nms, {boxes, scores, max_out, iou_tresh, score_tresh});
    auto g_nms = ov::as_type_ptr<ov::op::v13::NMSRotated>(builder.create());

    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type_attr(), nms->get_output_type_attr());
    EXPECT_EQ(g_nms->get_clockwise(), nms->get_clockwise());
}
