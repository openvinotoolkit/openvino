// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/parameter.hpp>
#include <openvino/op/proposal.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;

TEST(StaticShapeInferenceTest, ProposalV0Test) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto op = std::make_shared<op::v0::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    const std::vector<ov::StaticShape> input_shapes = {ov::StaticShape{batch_size, 12, 34, 62},
                                                       ov::StaticShape{batch_size, 24, 34, 62},
                                                       ov::StaticShape{3}};
    std::vector<ov::StaticShape> output_shapes = {ov::StaticShape{}};
    shape_inference(op.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{batch_size * attrs.post_nms_topn, 5}));
}

TEST(StaticShapeInferenceTest, ProposalV4Test) {
    op::v0::Proposal::Attributes attrs;
    attrs.base_size = 1;
    attrs.pre_nms_topn = 20;
    attrs.post_nms_topn = 200;
    const size_t batch_size = 7;

    auto class_probs = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto class_bbox_deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto image_shape = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto op = std::make_shared<op::v4::Proposal>(class_probs, class_bbox_deltas, image_shape, attrs);
    const std::vector<ov::StaticShape> input_shapes = {ov::StaticShape{batch_size, 12, 34, 62},
                                                       ov::StaticShape{batch_size, 24, 34, 62},
                                                       ov::StaticShape{3}};
    std::vector<ov::StaticShape> output_shapes = {ov::StaticShape{}, ov::StaticShape{}};
    shape_inference(op.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{batch_size * attrs.post_nms_topn, 5}));
    ASSERT_EQ(output_shapes[1], (StaticShape{batch_size * attrs.post_nms_topn}));
}