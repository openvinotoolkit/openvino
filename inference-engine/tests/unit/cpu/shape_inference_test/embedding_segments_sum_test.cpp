// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <embedding_segments_sum_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/embedding_segments_sum.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils.hpp"
#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, EmbeddingSegmentsSum) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::v0::Parameter>(element::i64, Shape{4});
    auto num_segments = op::v0::Constant::create(element::i64, ov::Shape{}, {3});
    auto default_index = make_shared<op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, Shape{4});

    auto ess = make_shared<op::v3::EmbeddingSegmentsSum>(emb_table,
                                                         indices,
                                                         segment_ids,
                                                         num_segments,
                                                         default_index,
                                                         per_sample_weights);

    std::vector<ov::PartialShape> input_shapes = {ov::PartialShape{5, 2},
                                                  ov::PartialShape{4},
                                                  ov::PartialShape{4},
                                                  ov::PartialShape{},
                                                  ov::PartialShape{},
                                                  ov::PartialShape{4}};
    std::vector<ov::PartialShape> output_shapes = {PartialShape::dynamic()};
    shape_infer(ess.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes.size(), 1);
    ASSERT_EQ(output_shapes[0], ov::PartialShape({3, 2}));

    check_static_shape(ess, {Shape{5, 2}, Shape{4}, Shape{4}, Shape{}, Shape{}, Shape{4}}, {StaticShape{3, 2}});

    check_static_shape(ess, {Shape{5, 2}, Shape{4}, Shape{4}, 8, Shape{}, Shape{4}}, {StaticShape{8, 2}});
}
