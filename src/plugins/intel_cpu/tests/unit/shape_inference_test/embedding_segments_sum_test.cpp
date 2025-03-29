// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_segments_sum_shape_inference.hpp"
#include "gmock/gmock.h"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using namespace ov::opset10;
using namespace testing;
using namespace std;

class EmbeddingSegmentsSumV3StaticShapeInferenceTest : public OpStaticShapeInferenceTest<op::v3::EmbeddingSegmentsSum> {
protected:
    void SetUp() override {
        output_shapes.resize(1);
    }
};

TEST_F(EmbeddingSegmentsSumV3StaticShapeInferenceTest, default_ctor) {
    const auto op = make_op();

    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{4}, StaticShape{}, StaticShape{}, StaticShape{4}};

    int64_t num_segments = 4;
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{3, {element::i64, ov::Shape{}, &num_segments}}};

    output_shapes = shape_inference(op.get(), input_shapes, const_map);
    EXPECT_EQ(output_shapes[0], (StaticShape{4, 2, 6}));
}

TEST_F(EmbeddingSegmentsSumV3StaticShapeInferenceTest, constant_input) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1});
    auto indices = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto segment_ids = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto num_segments = op::v0::Constant::create(element::i64, ov::Shape{}, {3});
    auto default_index = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1});

    auto op = make_op(emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{4}, StaticShape{}, StaticShape{}, StaticShape{4}},
    output_shapes = shape_inference(op.get(), input_shapes);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 2, 6}));
}

TEST_F(EmbeddingSegmentsSumV3StaticShapeInferenceTest, constant_map) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1});
    auto indices = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto segment_ids = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto num_segments = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});
    auto default_index = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1});

    auto op = make_op(emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);
    input_shapes = {StaticShape{5, 2, 6}, StaticShape{4}, StaticShape{4}, StaticShape{}, StaticShape{}, StaticShape{4}};

    int64_t num_segm_val = 3;
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{3, {element::i64, ov::Shape{}, &num_segm_val}}};

    output_shapes = shape_inference(op.get(), input_shapes, const_map);
    EXPECT_EQ(output_shapes[0], (StaticShape{3, 2, 6}));
}

TEST_F(EmbeddingSegmentsSumV3StaticShapeInferenceTest, basic) {
    auto emb_table = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1, -1});
    auto indices = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto segment_ids = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{-1});
    auto num_segments = op::v0::Constant::create(element::i64, ov::Shape{}, {3});
    auto default_index = make_shared<op::v0::Parameter>(element::i64, ov::PartialShape{});
    auto per_sample_weights = make_shared<op::v0::Parameter>(element::f32, ov::PartialShape{-1});

    auto op = make_op(emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights);

    output_shapes = shape_inference(op.get(), StaticShapeVector{{5, 2}, {4}, {4}, {}, {}, {4}});
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{3, 2}));

    int64_t num_segm_val = 8;
    const auto const_map = std::unordered_map<size_t, ov::Tensor>{{3, {element::i64, ov::Shape{}, &num_segm_val}}};
    output_shapes = shape_inference(op.get(), StaticShapeVector{{5, 2}, {4}, {4}, {}, {}, {4}}, const_map);
    EXPECT_THAT(output_shapes, ElementsAre(StaticShape{8, 2}));
}
