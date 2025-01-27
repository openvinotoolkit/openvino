// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/add.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::op::v1::ReduceMax;
using ov::op::v1::Add;
using testing::HasSubstr;

class TypePropSegmentMaxTest : public TypePropOpTest<op::v16::SegmentMax> {};

TEST_F(TypePropSegmentMaxTest, default_ctor) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});

    const auto op = make_op(data, segment_ids, 0);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_empty_segment_value(), 0);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 225}));
}

TEST_F(TypePropSegmentMaxTest, non_default_args_no_values) {
    PartialShape data_shape{3, 12, 81};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(nullptr, data_symbols[1], data_symbols[2]));
}

TEST_F(TypePropSegmentMaxTest, incorrect_inputs) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape{2});
    const auto num_segments = std::make_shared<Parameter>(element::i32, PartialShape{});
    {
        const auto num_segments_i64 = std::make_shared<Parameter>(element::i64, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids, num_segments_i64, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("The element types of the segment_ids and num_segments tensors must match."));
    }
    {
        const auto segment_ids_f32 = std::make_shared<Parameter>(element::f32, PartialShape{2});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_f32, num_segments, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the segment_ids input be i32 or i64."));
    }
    {
        const auto segment_ids_nd = std::make_shared<Parameter>(element::i32, PartialShape{2, 3});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_nd, num_segments, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("segment_ids must be a 1D input."));
    }
    {
        const auto num_segments_nd = std::make_shared<Parameter>(element::i32, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids, num_segments_nd, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("num_segments must be a scalar input."));
    }
    {
        const auto segment_ids_unsorted = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{1, 0, 1});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_unsorted, num_segments, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("segment_ids must be sorted."));
    }
    {
        const auto data_scalar = std::make_shared<Parameter>(element::i32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(data_scalar, segment_ids, num_segments, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("The data input cannot be a scalar."));
    }
    {
        const auto segment_ids_short = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 0});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_short, num_segments, 0),
                        ov::NodeValidationFailure,
                        HasSubstr("The number of elements in segment_ids must match the first dimension of data."));
    }
}

TEST_F(TypePropSegmentMaxTest, num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, dynamic_num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, interval_num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape{{2, 4}});
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, all_inputs_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape::dynamic()));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, all_inputs_interval) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{2, 4}, {12, 15}, {18, 300}});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{{2, 4}});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), {12, 15}, {18, 300}}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, data_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{6});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape::dynamic()));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, num_segments_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, dynamic_dimensions) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}

TEST_F(TypePropSegmentMaxTest, dynamic_num_segments_const_segment_ids) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto num_segments = std::make_shared<Parameter>(element::i32, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, 20);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{3, Dimension::dynamic(), 81}));
    EXPECT_EQ(op->get_empty_segment_value(), 20);
}
}  // namespace test
}  // namespace ov
