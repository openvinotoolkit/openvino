//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes)
{
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);
    ASSERT_EQ(G->get_output_element_type(0), element::i32);
    ASSERT_EQ(G->get_output_element_type(1), element::i32);
    ASSERT_EQ(G->get_output_shape(0), out_shape1);
    ASSERT_EQ(G->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes_with_bi)
{
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto BI = op::Constant::create(element::i32, Shape{}, {1});
    auto G =
        make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, BI, false, element::i64, element::i64);
    ASSERT_EQ(G->get_output_element_type(0), element::i64);
    ASSERT_EQ(G->get_output_element_type(1), element::i64);
    ASSERT_EQ(G->get_output_shape(0), out_shape1);
    ASSERT_EQ(G->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes_with_dinemic_bi)
{
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto BI = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto G =
        make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, BI, false, element::i64, element::i64);
    ASSERT_EQ(G->get_output_element_type(0), element::i64);
    ASSERT_EQ(G->get_output_element_type(1), element::i64);
    ASSERT_EQ(G->get_output_shape(0), out_shape1);
    ASSERT_EQ(G->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_output_static_shape1)
{
    PartialShape logits_shape{Dimension::dynamic(), 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
    ASSERT_EQ(G->get_output_element_type(0), element::i32);
    ASSERT_EQ(G->get_output_element_type(1), element::i32);
    ASSERT_EQ(G->get_output_shape(0), out_shape1);
    ASSERT_EQ(G->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_shapes)
{
    PartialShape logits_shape{Dimension::dynamic(), Dimension::dynamic(), 1200};
    PartialShape seq_len_shape{Dimension::dynamic()};
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
    ASSERT_EQ(G->get_output_element_type(0), element::i32);
    ASSERT_EQ(G->get_output_element_type(1), element::i32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape1));
    ASSERT_TRUE(G->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_ranks1)
{
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_len_shape{Dimension::dynamic()};
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);
    ASSERT_EQ(G->get_output_element_type(0), element::i32);
    ASSERT_EQ(G->get_output_element_type(1), element::i32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape1));
    ASSERT_TRUE(G->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_ranks2)
{
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_mask_shape = PartialShape::dynamic();
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_mask_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
    ASSERT_EQ(G->get_output_element_type(0), element::i32);
    ASSERT_EQ(G->get_output_element_type(1), element::i32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape1));
    ASSERT_TRUE(G->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_incorrect_rank)
{
    PartialShape logits_shape{Dimension::dynamic(), 100, 1200, 5};
    PartialShape seq_len_shape{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);

    try
    {
        auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The rank of logits tensor must be equal to 3."));
    }
    catch (...)
    {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_seq_len_incorrect_rank2)
{
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3, 100};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_len_shape);

    try
    {
        auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The rank of sequence len tensor must be equal to 1."));
    }
    catch (...)
    {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_seq_len_mismatched_dim1)
{
    PartialShape logits_shape{4, 100, 1200};
    PartialShape seq_mask_shape{3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::i32, seq_mask_shape);

    try
    {
        auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The first dimensions of input tensors must match."));
    }
    catch (...)
    {
        FAIL() << "Rank check failed for unexpected reason";
    }
}
