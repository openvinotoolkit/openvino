// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <embeddingbag_offsets_shape_inference.hpp>
#include <openvino/core/coordinate_diff.hpp>
#include <openvino/op/embeddingbag_offsets_sum.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>

#include "utils/shape_inference/static_shape.hpp"

using namespace ov;
using namespace std;

TEST(StaticShapeInferenceTest, EmbeddingBagOffsetsSumV3) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    auto indices = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto offsets = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto default_index = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape::dynamic());
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());

    auto ebos =
        make_shared<op::v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index, per_sample_weights);

    std::vector<ov::PartialShape> input_shapes = {ov::PartialShape{5, 2},
                                                  ov::PartialShape{4},
                                                  ov::PartialShape{3},
                                                  ov::PartialShape{},
                                                  ov::PartialShape{4}};
    std::vector<ov::PartialShape> output_shapes = {PartialShape::dynamic()};
    shape_infer(ebos.get(), input_shapes, output_shapes);
    ASSERT_EQ(output_shapes.size(), 1);
    ASSERT_EQ(output_shapes[0], ov::PartialShape({3, 2}));

    std::vector<StaticShape> static_input_shapes = {StaticShape{5, 2},
                                                    StaticShape{4},
                                                    StaticShape{3},
                                                    StaticShape{},
                                                    StaticShape{4}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{}};
    shape_infer(ebos.get(), static_input_shapes, static_output_shapes);

    ASSERT_EQ(static_output_shapes.size(), 1);
    ASSERT_EQ(static_output_shapes[0], StaticShape({3, 2}));
}
