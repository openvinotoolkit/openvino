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

TEST(type_prop, ctc_greedy_decoder_static_shapes)
{
    PartialShape logits_shape{100, 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    Shape out_shape{3, 100, 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_output_static_shape1)
{
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    Shape out_shape{3, 100, 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_output_static_shape2)
{
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, Dimension::dynamic()};
    Shape out_shape{3, 100, 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_dynamic_shapes)
{
    PartialShape logits_shape{Dimension::dynamic(), Dimension::dynamic(), 1200};
    PartialShape seq_mask_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_dynamic_ranks1)
{
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_mask_shape{100, Dimension::dynamic()};
    PartialShape out_shape{Dimension::dynamic(), 100, 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_dynamic_ranks2)
{
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_mask_shape = PartialShape::dynamic();
    PartialShape out_shape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_incorrect_rank)
{
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200, 5};
    PartialShape seq_mask_shape{100, 3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);

    try
    {
        auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
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

TEST(type_prop, ctc_greedy_decoder_incorrect_rank2)
{
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, 3, 2};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);

    try
    {
        auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The rank of sequence mask tensor must be equal to 2."));
    }
    catch (...)
    {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_mismatched_dim1)
{
    PartialShape logits_shape{100, 4, 1200};
    PartialShape seq_mask_shape{100, 3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);

    try
    {
        auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("The second dimensions of input tensors must match."));
    }
    catch (...)
    {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_mismatched_dim2)
{
    PartialShape logits_shape{101, 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);

    try
    {
        auto G = make_shared<op::CTCGreedyDecoder>(P, I, false);
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
