// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/experimental_detectron_generate_proposals.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;
using ExperimentalProposals = op::v6::ExperimentalDetectronGenerateProposalsSingleImage;

TEST(StaticShapeInferenceTest, ExperimentalProposalsTest) {
    ExperimentalProposals::Attributes attrs;
    attrs.min_size = 0.0f;
    attrs.nms_threshold = 0.699999988079071f;
    attrs.post_nms_count = 1000;
    attrs.pre_nms_count = 1000;
    size_t post_nms_count = 1000;

    auto im_info = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1});
    auto anchors = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1});
    auto deltas = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto scores = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1});

    auto proposals = std::make_shared<ExperimentalProposals>(im_info, anchors, deltas, scores, attrs);

    const std::vector<StaticShape> input_shapes = {StaticShape{3},
                                                   StaticShape{201600, 4},
                                                   StaticShape{12, 200, 336},
                                                   StaticShape{3, 200, 336}};
    std::vector<StaticShape> output_shapes = {StaticShape{}, StaticShape{}};
    shape_inference(proposals.get(), input_shapes, output_shapes);

    ASSERT_EQ(output_shapes[0], (StaticShape{post_nms_count, 4}));
    ASSERT_EQ(output_shapes[1], (StaticShape{post_nms_count}));
}