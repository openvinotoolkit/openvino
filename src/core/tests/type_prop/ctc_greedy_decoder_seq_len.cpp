// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/ops.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, ctc_greedy_decoder_seq_len_default_ctor) {
    PartialShape logits_shape{2, 10, 1200};
    PartialShape seq_len_shape{2};

    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>();

    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::f32, seq_len_shape);
    op->set_arguments(OutputVector{logits_param, seq_len_param});

    op->set_merge_repeated(false);
    EXPECT_EQ(op->get_merge_repeated(), false);

    op->set_merge_repeated(true);
    EXPECT_EQ(op->get_merge_repeated(), true);

    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);

    op->set_classes_index_type(element::i64);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);

    op->set_sequence_length_type(element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::i64);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 10}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2}));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes) {
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_shape(0), out_shape1);
    EXPECT_EQ(op->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes_with_bi) {
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto bi = op::v0::Constant::create(element::i32, Shape{}, {1});
    auto op =
        make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, bi, false, element::i64, element::i64);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::i64);
    EXPECT_EQ(op->get_output_shape(0), out_shape1);
    EXPECT_EQ(op->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_static_shapes_with_dinemic_bi) {
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto bi = make_shared<op::v0::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto op =
        make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, bi, false, element::i64, element::i64);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_element_type(1), element::i64);
    EXPECT_EQ(op->get_output_shape(0), out_shape1);
    EXPECT_EQ(op->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_output_static_shape1) {
    PartialShape logits_shape{Dimension::dynamic(), 100, 1200};
    PartialShape seq_len_shape{3};
    Shape out_shape1{3, 100};
    Shape out_shape2{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_shape(0), out_shape1);
    EXPECT_EQ(op->get_output_shape(1), out_shape2);
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_shapes) {
    PartialShape logits_shape{Dimension::dynamic(), Dimension::dynamic(), 1200};
    PartialShape seq_len_shape{Dimension::dynamic()};
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape1));
    EXPECT_TRUE(op->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_ranks1) {
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_len_shape{Dimension::dynamic()};
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape1));
    EXPECT_TRUE(op->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_dynamic_ranks2) {
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_len_shape = PartialShape::dynamic();
    PartialShape out_shape1{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape2{Dimension::dynamic()};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape1));
    EXPECT_TRUE(op->get_output_partial_shape(1).same_scheme(out_shape2));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_interval_symboled_dims_all) {
    PartialShape logits_shape{{2, 6}, {10, 100}, {600, 1200}};
    PartialShape seq_len_shape{{4, 8}};

    auto l_symbols = set_shape_symbols(logits_shape);
    auto s_symbols = set_shape_symbols(seq_len_shape);

    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::f32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param);

    // Output 0
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 6}, {10, 100}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(l_symbols[0], l_symbols[1]));

    // Output 1
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{4, 6}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(l_symbols[0]));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_interval_symboled_dims_in0) {
    PartialShape logits_shape{{2, 6}, {10, 100}, {600, 1200}};
    PartialShape seq_len_shape{{4, 8}};

    auto symbols = set_shape_symbols(logits_shape);

    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::f32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param);

    // Output 0
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 6}, {10, 100}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], symbols[1]));

    // Output 1
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{4, 6}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0]));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_interval_symboled_dims_in1) {
    PartialShape logits_shape{{2, 6}, {10, 100}, {600, 1200}};
    PartialShape seq_len_shape{{4, 8}};

    auto symbols = set_shape_symbols(seq_len_shape);

    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::f32, seq_len_shape);
    auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param);

    // Output 0
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{{4, 6}, {10, 100}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(symbols[0], nullptr));

    // Output 1
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{{4, 6}}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)), ElementsAre(symbols[0]));
}

TEST(type_prop, ctc_greedy_decoder_seq_len_incorrect_rank) {
    PartialShape logits_shape{Dimension::dynamic(), 100, 1200, 5};
    PartialShape seq_len_shape{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);

    try {
        auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The rank of logits tensor must be equal to 3."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_seq_len_incorrect_rank2) {
    PartialShape logits_shape{3, 100, 1200};
    PartialShape seq_len_shape{3, 100};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);

    try {
        auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The rank of sequence len tensor must be equal to 1."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_seq_len_mismatched_dim1) {
    PartialShape logits_shape{4, 100, 1200};
    PartialShape seq_len_shape{3};
    auto logits_param = make_shared<op::v0::Parameter>(element::f32, logits_shape);
    auto seq_len_param = make_shared<op::v0::Parameter>(element::i32, seq_len_shape);

    try {
        auto op = make_shared<op::v6::CTCGreedyDecoderSeqLen>(logits_param, seq_len_param, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The first dimensions of input tensors must match."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}
