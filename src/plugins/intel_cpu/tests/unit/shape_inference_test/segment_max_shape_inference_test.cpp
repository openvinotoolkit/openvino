// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::intel_cpu;
using ov::op::v0::Constant, ov::op::v0::Parameter;
using testing::HasSubstr;

struct SegmentMaxTestParams {
    ov::Shape data_shape;
    std::vector<int32_t> segment_ids_val;
    int64_t num_segments_val;
    ov::Shape expected_output_shape;
};

class SegmentMaxStaticTestSuite : public ::testing::TestWithParam<std::tuple<SegmentMaxTestParams, bool>> {};

class SegmentMaxStaticShapeInferenceTest: public OpStaticShapeInferenceTest<op::v16::SegmentMax> {};

TEST_F(SegmentMaxStaticShapeInferenceTest, segment_ids_from_tensor_accessor) {
    const auto data = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto op = make_op(data, segment_ids, ov::op::FillMode::ZERO);

    int64_t segment_ids_val[] = {0, 0, 1, 1, 4, 4, 5, 5, 5, 5};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{1, {element::i64, ov::Shape{10}, segment_ids_val}}};

    const auto input_shapes = StaticShapeVector{{10, 12, 289}, {10}, {}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({6, 12, 289}));
}

TEST_F(SegmentMaxStaticShapeInferenceTest, segment_ids_from_tensor_accessor_with_num_segments) {
    const auto data = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto num_segments = std::make_shared<Parameter>(element::i64, ov::PartialShape::dynamic());
    const auto op = make_op(data, segment_ids, num_segments, ov::op::FillMode::ZERO);

    int64_t segment_ids_val[] = {0, 0, 1, 1, 4, 4, 5, 5, 5, 5};
    int64_t num_segments_val[] = {6};
    auto const_inputs = std::unordered_map<size_t, Tensor>{{1, {element::i64, ov::Shape{10}, segment_ids_val}},
                                                           {2, {element::i64, ov::Shape{}, num_segments_val}}};

    const auto input_shapes = StaticShapeVector{{5, 12, 289}, {5}, {}};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor(const_inputs));
    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape({6, 12, 289}));
}

TEST_P(SegmentMaxStaticTestSuite, SegmentMaxStaticShapeInference) {
    const auto& param = std::get<0>(GetParam());
    const bool use_num_segments = std::get<1>(GetParam());
    const auto& data_shape = param.data_shape;
    const auto& segment_ids_val = param.segment_ids_val;
    const auto& expected_output_shape = param.expected_output_shape;
    const auto segment_ids_shape = ov::Shape{segment_ids_val.size()};

    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto segment_ids = std::make_shared<Constant>(element::i32, segment_ids_shape, segment_ids_val);

    std::shared_ptr<op::v16::SegmentMax> op;
    if (use_num_segments) {
        const auto& num_segments_val = param.num_segments_val;
        const auto num_segments = std::make_shared<Constant>(element::i32, ov::Shape{}, num_segments_val);
        op = std::make_shared<op::v16::SegmentMax>(data, segment_ids, num_segments, ov::op::FillMode::ZERO);
    } else {
        op = std::make_shared<op::v16::SegmentMax>(data, segment_ids, ov::op::FillMode::ZERO);
    }

    const auto input_shapes = use_num_segments
                                  ? StaticShapeVector{data_shape, segment_ids_shape, {}}
                                  : StaticShapeVector{data_shape, segment_ids_shape};
    auto shape_infer = make_shape_inference(op);
    const auto input_shape_refs = make_static_shape_refs(input_shapes);
    const auto output_shapes = *shape_infer->infer(input_shape_refs, make_tensor_accessor());

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), StaticShape(expected_output_shape));
}

INSTANTIATE_TEST_SUITE_P(SegmentMaxStaticShapeInferenceTests,
                         SegmentMaxStaticTestSuite,
                         ::testing::Combine(
                             ::testing::Values(SegmentMaxTestParams{ov::Shape{6},                                    // data shape
                                                                    std::vector<int32_t>{0, 0, 1, 4, 4, 4},          // segment_ids values
                                                                    5,                                               // num_segments value
                                                                    ov::Shape{5}},                                   // expected output shape
                                               SegmentMaxTestParams{ov::Shape{1, 23},                                // data shape
                                                                    std::vector<int32_t>{200},                       // segment_ids values
                                                                    201,                                             // num_segments value
                                                                    ov::Shape{201, 23}},                             // expected output shape
                                               SegmentMaxTestParams{ov::Shape{4, 23},                                // data shape
                                                                    std::vector<int32_t>{0, 0, 0, 0},                // segment_ids values
                                                                    1,                                               // num_segments value
                                                                    ov::Shape{1, 23}},                               // expected output shape
                                               SegmentMaxTestParams{ov::Shape{2, 6, 24, 1},                          // data shape
                                                                    std::vector<int32_t>{2, 6},                      // segment_ids values
                                                                    7,                                               // num_segments value
                                                                    ov::Shape{7, 6, 24, 1}}),                        // expected output shape
                             ::testing::Bool()));                                                                    // use_num_segments
