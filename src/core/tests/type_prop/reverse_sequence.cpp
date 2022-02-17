// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, reverse_sequence_default_attributes) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{4});
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths);

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
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 2, 3, 4, 5});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

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
            auto data = std::make_shared<op::Parameter>(et, PartialShape{4, 4});
            auto seq_lengths = std::make_shared<op::Parameter>(element::i32, PartialShape{4});

            EXPECT_NO_THROW(const auto unused = std::make_shared<op::ReverseSequence>(data, seq_lengths));
        } catch (...) {
            FAIL() << "Data input element type validation check failed for unexpected reason";
        }
    }
}

TEST(type_prop, reverse_sequence_invalid_seq_lengths_et) {
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    std::vector<element::Type> invalid_et{element::bf16, element::f16, element::f32, element::boolean};
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::boolean, PartialShape{4});
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Invalid element type of seq_lengths input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Sequence lengths element type must be numeric type."));
    } catch (...) {
        FAIL() << "Element type validation check of seq_lengths input failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_invalid_data_rank) {
    std::vector<PartialShape> invalid_pshapes = {{}, {4}};
    for (auto& pshape : invalid_pshapes) {
        try {
            auto data = make_shared<op::Parameter>(element::f32, pshape);
            auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{1});
            auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths);
            FAIL() << "Invalid rank of data input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Data input rank should be equal or greater than 2."));
        } catch (...) {
            FAIL() << "Rank check of data input failed for unexpected reason";
        }
    }
}

TEST(type_prop, reverse_sequence_invalid_seq_lengths_rank) {
    std::vector<PartialShape> invalid_pshapes = {{}, {4, 1}, {1, 1, 4}};
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    for (auto& pshape : invalid_pshapes) {
        try {
            auto seq_lengths = make_shared<op::Parameter>(element::i32, pshape);
            auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
            FAIL() << "Invalid rank of seq_lengths input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), std::string("Sequence lengths rank must be equal to 1."));
        } catch (...) {
            FAIL() << "Rank check of seq_lengths input failed for unexpected reason";
        }
    }
}

TEST(type_prop, reverse_sequence_invalid_batch_axis_value) {
    int64_t batch_axis = 3;
    int64_t seq_axis = 1;
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Invalid value of batch_axis attribute not detected";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 3 out of the tensor rank"));
    } catch (...) {
        FAIL() << "Out of bounds check of batch_axis attribute failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_invalid_seq_axis_value) {
    int64_t batch_axis = 1;
    int64_t seq_axis = 3;
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Invalid value of seq_axis attribute not detected";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 3 out of the tensor rank"));
    } catch (...) {
        FAIL() << "Out of bounds check of seq_axis attribute failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_incompatible_seq_len_size_with_batch_dim) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{4, 3, 2});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths);
        FAIL() << "Incompatible number of elements between seq_lengths and batch dimension of data input not detected";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence lengths input size (3) is not equal to batch axis dimension of data input (4)"));
    } catch (...) {
        FAIL() << "Number of elements of input seq_lengths check with respect batch dimension of input failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_dynamic_inputs_with_dynamic_rank) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    // Unrealistic values, but they don't matter here.
    int64_t batch_axis = 202;
    int64_t seq_axis = 909;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_dynamic_rank) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    // Unrealistic values, but they don't matter here.
    int64_t batch_axis = 202;
    int64_t seq_axis = 909;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_EQ(reverse_seq->get_output_partial_shape(0), (PartialShape::dynamic()));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_seq_lenghts_input_dynamic_rank) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 4, 6, 8});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(reverse_seq->get_output_partial_shape(0).same_scheme(PartialShape{2, 4, 6, 8}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_and_seq_lengths_dynamic_rank) {
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape::dynamic());
    int64_t batch_axis = 0;
    int64_t seq_axis = 1;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(reverse_seq->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_invalid_batch_axis) {
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 4;
    int64_t seq_axis = 1;
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Invalid batch_axis attribute value not detected (rank-static dynamic shape)";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 4 out of the tensor rank"));
    } catch (...) {
        FAIL() << "Out of bounds check of batch_axis attribute failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_dynamic_invalid_seq_axis) {
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 1;
    int64_t seq_axis = 4;
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Invalid seq_axis attribute value not detected (rank-static dynamic shape)";
    } catch (const ngraph_error& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Parameter axis 4 out of the tensor rank"));
    } catch (...) {
        FAIL() << "Out of bounds check of seq_axis attribute failed for unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank) {
    auto data = make_shared<op::Parameter>(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(reverse_seq->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_seq_lengths_input_dynamic_rank) {
    auto data =
        make_shared<op::Parameter>(element::f32,
                                   PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(reverse_seq->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_data_input_static_rank_with_static_batch_dim) {
    auto data =
        make_shared<op::Parameter>(element::f32,
                                   PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{3});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);

    EXPECT_TRUE(reverse_seq->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()}));
    EXPECT_EQ(reverse_seq->get_output_element_type(0), element::f32);
}

TEST(type_prop, reverse_sequence_dynamic_incompatible_data_input_static_rank_with_static_batch_dim) {
    auto data =
        make_shared<op::Parameter>(element::f32,
                                   PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, Dimension::dynamic()});
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{4});
    int64_t batch_axis = 2;
    int64_t seq_axis = 1;
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Incompatible number of elements between seq_lengths and batch dimension of data input not detected "
                  "(rank-static dynamic shape)";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Sequence lengths input size (4) is not equal to batch axis dimension of data input (3)"));
    } catch (...) {
        FAIL() << "Number of elements of input seq_lengths check with respect batch dimension of input failed for "
                  "unexpected reason";
    }
}

TEST(type_prop, reverse_sequence_dynamic_invalid_negative_axis_and_data_input_dynamic_rank) {
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto seq_lengths = make_shared<op::Parameter>(element::i32, PartialShape{1});
    int64_t batch_axis = 1;
    int64_t seq_axis = -2;
    try {
        auto reverse_seq = make_shared<op::ReverseSequence>(data, seq_lengths, batch_axis, seq_axis);
        FAIL() << "Dynamic rank of data input for negative axis not detected";
    } catch (const CheckFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Rank must be static in order to normalize negative axis=-2"));
    } catch (...) {
        FAIL() << "Static rank of data input for negative axis validation check failed for unexpected reason";
    }
}
