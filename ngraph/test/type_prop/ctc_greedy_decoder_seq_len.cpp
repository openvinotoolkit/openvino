//*****************************************************************************
// Copyright 2020 Intel Corporation
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
    Shape out_shape{3, 100};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_output_static_shape1)
{
    PartialShape logits_shape{Dimension::dynamic(), 100, 1200};
    PartialShape seq_mask_shape{3};
    Shape out_shape{3, 100};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_mask_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I, false);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_EQ(G->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_ranks1)
{
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_len_shape{Dimension::dynamic()};
    PartialShape out_shape{Dimension::dynamic(), Dimension::dynamic()};
    auto P = make_shared<op::Parameter>(element::f32, logits_shape);
    auto I = make_shared<op::Parameter>(element::f32, seq_len_shape);
    auto G = make_shared<op::v6::CTCGreedyDecoderSeqLen>(P, I);
    ASSERT_EQ(G->get_element_type(), element::f32);
    ASSERT_TRUE(G->get_output_partial_shape(0).same_scheme(out_shape));
}
