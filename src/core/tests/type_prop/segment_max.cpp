// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter, op::v1::Add, op::v1::ReduceMax, op::v1::StridedSlice, op::v3::ShapeOf;
using testing::HasSubstr;

class TypePropSegmentMaxTest : public TypePropOpTest<op::v16::SegmentMax> {};

TEST_F(TypePropSegmentMaxTest, default_ctor) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});

    const auto op = make_op(data, segment_ids, op::FillMode::LOWEST);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_fill_mode(), op::FillMode::LOWEST);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 225}));
}

TEST_F(TypePropSegmentMaxTest, non_default_args_no_values) {
    PartialShape data_shape{3, 12, 81};
    auto data_symbols = set_shape_symbols(data_shape);
    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::LOWEST);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::LOWEST);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(nullptr, data_symbols[1], data_symbols[2]));
}

TEST_F(TypePropSegmentMaxTest, num_segments_bigger_than_segment_ids) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{0, 0, 1});
    const auto num_segments = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int64_t>{40});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::LOWEST);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{40, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::LOWEST);
}

TEST_F(TypePropSegmentMaxTest, incorrect_inputs) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{3, 12, 225});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto num_segments = std::make_shared<Parameter>(element::i32, PartialShape{});
    {
        const auto num_segments_f32 = std::make_shared<Parameter>(element::f32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids, num_segments_f32, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the num_segments input be i32 or i64."));
    }
    {
        const auto segment_ids_f32 = std::make_shared<Parameter>(element::f32, PartialShape{3});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_f32, num_segments, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("The element type of the segment_ids input be i32 or i64."));
    }
    {
        const auto segment_ids_nd = std::make_shared<Parameter>(element::i32, PartialShape{2, 3});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_nd, num_segments, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("segment_ids must be a 1D input."));
    }
    {
        const auto num_segments_nd = std::make_shared<Parameter>(element::i32, PartialShape{1});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids, num_segments_nd, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("num_segments must be a scalar input."));
    }
    {
        const auto segment_ids_unsorted =
            std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{1, 0, 1});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_unsorted, num_segments, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("segment_ids must be sorted."));
    }
    {
        const auto data_scalar = std::make_shared<Parameter>(element::i32, PartialShape{});
        OV_EXPECT_THROW(std::ignore = make_op(data_scalar, segment_ids, num_segments, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("The data input cannot be a scalar."));
    }
    {
        const auto segment_ids_short = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 0});
        OV_EXPECT_THROW(std::ignore = make_op(data, segment_ids_short, num_segments, op::FillMode::LOWEST),
                        ov::NodeValidationFailure,
                        HasSubstr("The number of elements in segment_ids must match the first dimension of data."));
    }
}

TEST_F(TypePropSegmentMaxTest, num_segments_from_graph_shapeof) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto some_subgraph_result = std::make_shared<Parameter>(element::i32, PartialShape{8});
    const auto shape_of = std::make_shared<ShapeOf>(some_subgraph_result);
    const auto num_segments = std::make_shared<ReduceMax>(shape_of, Constant::create(element::i64, Shape{}, {0}));

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::LOWEST);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{8, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::LOWEST);
}

TEST_F(TypePropSegmentMaxTest, num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::LOWEST);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::LOWEST);
}

TEST_F(TypePropSegmentMaxTest, dynamic_num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, interval_num_segments_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape{{2, 4}});
    const auto max_segment_id = std::make_shared<ReduceMax>(segment_ids, Constant::create(element::i32, Shape{}, {0}));
    const auto num_segments = std::make_shared<Add>(max_segment_id, Constant::create(element::i32, Shape{}, {1}));

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids_input =
        std::make_shared<Constant>(element::i32, Shape{5}, std::vector<int32_t>{0, 1, 2, 3, 4});
    const auto begin = Constant::create(element::i64, Shape{1}, {0});
    const auto end = Constant::create(element::i64, Shape{1}, {3});
    const auto stride = Constant::create(element::i64, Shape{1}, {1});
    const auto segment_ids = std::make_shared<op::v1::StridedSlice>(segment_ids_input,
                                                                    begin,
                                                                    end,
                                                                    stride,
                                                                    std::vector<int64_t>{0},
                                                                    std::vector<int64_t>{0});

    const auto op = make_op(data, segment_ids, op::FillMode::ZERO);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, dynamic_segment_ids_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids_input = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto begin = Constant::create(element::i64, Shape{1}, {0});
    const auto end = Constant::create(element::i64, Shape{1}, {3});
    const auto stride = Constant::create(element::i64, Shape{1}, {1});
    const auto segment_ids = std::make_shared<op::v1::StridedSlice>(segment_ids_input,
                                                                    begin,
                                                                    end,
                                                                    stride,
                                                                    std::vector<int64_t>{0},
                                                                    std::vector<int64_t>{0});

    const auto op = make_op(data, segment_ids, op::FillMode::ZERO);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, interval_segment_ids_from_graph) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids_input = std::make_shared<Parameter>(element::i32, PartialShape{{2, 4}});
    const auto begin = Constant::create(element::i64, Shape{1}, {0});
    const auto end = Constant::create(element::i64, Shape{1}, {3});
    const auto stride = Constant::create(element::i64, Shape{1}, {1});
    const auto segment_ids = std::make_shared<op::v1::StridedSlice>(segment_ids_input,
                                                                    begin,
                                                                    end,
                                                                    stride,
                                                                    std::vector<int64_t>{0},
                                                                    std::vector<int64_t>{0});

    const auto op = make_op(data, segment_ids, op::FillMode::ZERO);
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, all_inputs_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, all_inputs_interval) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{{2, 4}, {12, 15}, {18, 300}});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{{2, 4}});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), {12, 15}, {18, 300}}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, data_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{6});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_const_num_segments_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_dynamic_num_segments_const) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto num_segments = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{7});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{7, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, const_num_segments_zero) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto num_segments = std::make_shared<Constant>(element::i32, Shape{}, std::vector<int32_t>{0});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{0, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, segment_ids_const_no_num_segments) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});

    const auto op = make_op(data, segment_ids, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, num_segments_dynamic) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, 12, 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 12, 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, dynamic_dimensions) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 81});
    const auto segment_ids = std::make_shared<Parameter>(element::i64, PartialShape{Dimension::dynamic()});
    const auto num_segments = std::make_shared<Parameter>(element::i64, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, dynamic_num_segments_const_segment_ids) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});
    const auto num_segments = std::make_shared<Parameter>(element::i32, PartialShape{});

    const auto op = make_op(data, segment_ids, num_segments, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic(), 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}

TEST_F(TypePropSegmentMaxTest, no_num_segments_const_segment_ids) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{3, Dimension::dynamic(), 81});
    const auto segment_ids = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int32_t>{0, 1, 2});

    const auto op = make_op(data, segment_ids, op::FillMode::ZERO);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, Dimension::dynamic(), 81}));
    EXPECT_EQ(op->get_fill_mode(), op::FillMode::ZERO);
}
}  // namespace ov::test
