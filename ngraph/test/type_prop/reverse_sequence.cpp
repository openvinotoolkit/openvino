// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, reverse_sequence_1_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{4, 4});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for seq_lengths whose rank isn't equal to 1";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Sequence indices must be a 1-dimensional tensor"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_batch_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 3;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds batch axis index";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 3 out of the tensor rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_sequence_index_oob)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 1;
        size_t seq_axis = 3;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw for out-of-bounds sequence axis index";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 3 out of the tensor rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_seq_len_size_equal_to_batch_dim)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, Shape{3});
    try
    {
        size_t batch_axis = 0;
        size_t seq_axis = 1;
        auto bc = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "ReverseSequence c-tor should throw when sequence length size isn't equal to "
                  "batch dimension";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence length (3) is not equal to batch axis dimension (4)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_partial_both_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Unrealistic values, but they don't matter here.
    size_t batch_axis = 202;
    size_t seq_axis = 909;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).is_dynamic());
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_left_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    // Unrealistic values, but they don't matter here.
    size_t batch_axis = 202;
    size_t seq_axis = 909;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).is_dynamic());
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_right_rank_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t batch_axis = 0;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    size_t batch_axis = 0;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_batch_axis_oob)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 4;
    size_t seq_axis = 1;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Batch axis out of bounds not detected (rank-static dynamic shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 4 out of the tensor rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_sequence_axis_oob)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 1;
    size_t seq_axis = 4;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Sequence axis out of bounds not detected (rank-static dynamic shape)";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 4 out of the tensor rank"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_dynamic)
{
    auto data = make_shared<op::Parameter>(element::f32,
                                           PartialShape{Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic(),
                                                        Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_partial_both_rank_static_dynamic_right_seq_length_dynamic)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic()});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(type_prop,
     reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_static)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(rs->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(rs->get_output_element_type(0), element::f32);
}

TEST(
    type_prop,
    reverse_sequence_partial_left_rank_static_dynamic_right_static_left_seq_length_static_inconsistent)
{
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{4});
    size_t batch_axis = 2;
    size_t seq_axis = 1;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Inconsistent sequence length not detected (rank-static dynamic shape)";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence length (4) is not equal to batch axis dimension (3)"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_negative_axis_dynamic_input_rank)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{1});
    int64_t batch_axis = 1;
    int64_t seq_axis = -2;
    try
    {
        auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Dynamic input rank for negative axis not detected";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Rank must be static in order to normalize negative axis=-2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_negative_axes_support)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4, 5});
    auto seq_lengths = make_shared<op::Parameter>(element::f32, PartialShape{3});
    int64_t batch_axis = -3;
    int64_t seq_axis = -2;

    auto rs = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(rs->get_batch_axis(), 2);
    EXPECT_EQ(rs->get_sequence_axis(), 3);
}
