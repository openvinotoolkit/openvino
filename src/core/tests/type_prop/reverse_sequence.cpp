// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset10.hpp"
#include "reverse_sequence_shape_inference.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace testing;

TEST(type_prop, reverse_sequence_default_attributes) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{4});
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths);

    EXPECT_EQ(reverse_seq->get_batch_axis(), 0);
    EXPECT_EQ(reverse_seq->get_sequence_axis(), 1);
    EXPECT_EQ(reverse_seq->get_input_size(), 2);
    EXPECT_EQ(reverse_seq->get_output_size(), 1);
    EXPECT_EQ(reverse_seq->get_output_element_type(0), (element::f32));
    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape{4, 3, 2}));
}

TEST(type_prop, reverse_sequence_negative_attribute_axes) {
    int64_t batch_axis = -3;
    int64_t seq_axis = -2;
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 3, 4, 5});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_batch_axis(), 2);
    EXPECT_EQ(reverse_seq->get_sequence_axis(), 3);
    EXPECT_EQ(reverse_seq->get_input_size(), 2);
    EXPECT_EQ(reverse_seq->get_output_size(), 1);
    EXPECT_EQ(reverse_seq->get_output_element_type(0), (element::f32));
    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape{1, 2, 3, 4, 5}));
}

TEST(type_prop, reverse_sequence_data_et) {
    std::vector<element::Type> element_types{element::u4,
                                             element::u8,
                                             element::u16,
                                             element::u32,
                                             element::i8,
                                             element::i16,
                                             element::i32,
                                             element::i64,
                                             element::f16,
                                             element::f32,
                                             element::boolean};
    for (auto& et : element_types) {
        try {
            auto data = std::make_shared<Parameter>(et, PartialShape{4, 4});
            auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{4});

            EXPECT_NO_THROW(const auto unused = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths));
        } catch (...) {
            FAIL() << "Data input element type validation check failed for unexpected reason";
        }
    }
}

TEST(type_prop, reverse_sequence_invalid_seq_lengths_et) {
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    std::vector<element::Type> invalid_et{element::bf16, element::f16, element::f32, element::boolean};
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::boolean, PartialShape{4});
    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        NodeValidationFailure,
        HasSubstr("Sequence lengths element type must be numeric type."));
}

TEST(type_prop, reverse_sequence_invalid_data_rank) {
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{1});

    for (auto& invalid_shape : PartialShapes{{}, {4}}) {
        auto data = std::make_shared<Parameter>(element::f32, invalid_shape);

        OV_EXPECT_THROW(auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths),
                        NodeValidationFailure,
                        HasSubstr("Data input rank should be equal or greater than 2."));
    }
}

TEST(type_prop, reverse_sequence_invalid_seq_lengths_rank) {
    constexpr int64_t batch_axis = 0, seq_axis = 1;
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});

    for (auto& invalid_shape : PartialShapes{{}, {4, 1}, {1, 1, 4}}) {
        auto seq_lengths = std::make_shared<Parameter>(element::i32, invalid_shape);

        OV_EXPECT_THROW(
            auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
            NodeValidationFailure,
            HasSubstr("Sequence lengths rank must be equal to 1."));
    }
}

TEST(type_prop, reverse_sequence_invalid_batch_axis_value) {
    int64_t batch_axis = 3;
    int64_t seq_axis = 1;
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        AssertFailure,
        HasSubstr("Axis 3 out of the tensor rank"));
}

TEST(type_prop, reverse_sequence_invalid_seq_axis_value) {
    int64_t batch_axis = 1;
    int64_t seq_axis = 3;
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        AssertFailure,
        HasSubstr("Axis 3 out of the tensor rank"));
}

TEST(type_prop, reverse_sequence_incompatible_seq_len_size_with_batch_dim) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths),
        NodeValidationFailure,
        HasSubstr("Sequence lengths input size (3) is not equal to batch axis dimension of data input (4)"));
}

TEST(type_prop, reverse_sequence_dynamic_inputs_with_dynamic_rank) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    // Unrealistic values, but they don't matter here.
    int64_t batch_axis = 202;
    int64_t seq_axis = 909;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_dynamic_rank) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});
    // Unrealistic values, but they don't matter here.
    int64_t batch_axis = 202;
    int64_t seq_axis = 909;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_seq_lenghts_input_dynamic_rank) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), PartialShape({2, 4, 6, 8}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_and_seq_lengths_dynamic_rank) {
    auto data = std::make_shared<Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_invalid_batch_axis) {
    auto data = std::make_shared<Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 4;
    int64_t seq_axis = 1;

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        AssertFailure,
        HasSubstr("Axis 4 out of the tensor rank"));
}

TEST(type_prop, reverse_sequence_dynamic_invalid_seq_axis) {
    auto data = std::make_shared<Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 1;
    int64_t seq_axis = 4;

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        AssertFailure,
        HasSubstr("Axis 4 out of the tensor rank"));
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank) {
    auto data_shape = PartialShape::dynamic(4);
    auto symbols = set_shape_symbols(data_shape);
    auto data = std::make_shared<Parameter>(element::f32, data_shape);
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), PartialShape({-1, -1, 3, -1}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
    EXPECT_THAT(get_shape_symbols(reverse_seq->get_output_partial_shape(0)), symbols);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_seq_lengths_input_dynamic_rank) {
    auto data =
        std::make_shared<Parameter>(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), PartialShape({-1, -1, 3, -1}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_with_static_batch_dim) {
    auto data =
        std::make_shared<Parameter>(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), PartialShape({-1, -1, 3, -1}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_incompatible_data_input_static_rank_with_static_batch_dim) {
    auto data =
        std::make_shared<Parameter>(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{4});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;

    OV_EXPECT_THROW(
        auto reverse_seq = std::make_shared<op::v0::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis),
        NodeValidationFailure,
        HasSubstr("Sequence lengths input size (4) is not equal to batch axis dimension of data input (3)"));
}

class TypePropReverseSequenceV0Test : public TypePropOpTest<op::v0::ReverseSequence> {};

TEST_F(TypePropReverseSequenceV0Test, dynamic_invalid_negative_axis_and_data_input_dynamic_rank) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{1});
    int64_t batch_axis = 1;
    int64_t seq_axis = -2;

    OV_EXPECT_THROW(auto reverse_seq = make_op(data, seq_lengths, batch_axis, seq_axis),
                    AssertFailure,
                    HasSubstr("Rank must be static in order to normalize negative axis: -2"));
}

TEST_F(TypePropReverseSequenceV0Test, default_ctor) {
    auto data = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto op = make_op();

    op->set_arguments(OutputVector{data, seq_lengths});
    op->set_batch_axis(1);
    op->set_sequence_axis(0);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_batch_axis(), 1);
    EXPECT_EQ(op->get_sequence_axis(), 0);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 3, 2}));
}

TEST_F(TypePropReverseSequenceV0Test, default_ctor_no_arguments) {
    auto op = make_op();
    op->set_batch_axis(1);

    const auto output_shapes = shape_infer(op.get(), PartialShapes{{{4, 5}, 3, 2}, {3}});

    EXPECT_EQ(output_shapes.size(), 1);
    EXPECT_EQ(output_shapes.front(), PartialShape({{4, 5}, 3, 2}));

    EXPECT_EQ(op->get_input_size(), 0);
    EXPECT_EQ(op->get_output_size(), 0);
}

TEST_F(TypePropReverseSequenceV0Test, data_shape_interval_and_sequence_static_dim_with_symbols) {
    auto data_shape = PartialShape{{2, 5}, 4, {1, 3}};
    auto seq_shape = PartialShape{3};
    auto data_symbols = set_shape_symbols(data_shape);
    set_shape_symbols(seq_shape);

    auto data = std::make_shared<Parameter>(element::f32, data_shape);
    auto seq_lengths = std::make_shared<Parameter>(element::i32, seq_shape);
    auto op = make_op(data, seq_lengths, 0, 1);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({3, 4, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], data_symbols[1], data_symbols[2]));
}

TEST_F(TypePropReverseSequenceV0Test, data_shape_and_sequence_interval_dim_with_symbols) {
    auto data_shape = PartialShape{{2, 5}, 4, {1, 3}};
    auto seq_shape = PartialShape{{3, 6}};
    auto data_symbols = set_shape_symbols(data_shape);
    set_shape_symbols(seq_shape);

    auto data = std::make_shared<Parameter>(element::f32, data_shape);
    auto seq_lengths = std::make_shared<Parameter>(element::i32, seq_shape);
    auto op = make_op(data, seq_lengths, -3, -1);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{3, 5}, 4, {1, 3}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(data_symbols[0], data_symbols[1], data_symbols[2]));
}
