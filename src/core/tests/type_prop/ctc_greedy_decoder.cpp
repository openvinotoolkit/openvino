// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace testing;

TEST(type_prop, ctc_greedy_decoder_default_constructor) {
    PartialShape data_shape{100, 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    PartialShape expected_shape{3, 100, 1, 1};

    auto op = make_shared<ov::op::v0::CTCGreedyDecoder>();

    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    op->set_arguments(OutputVector{data, seq_mask});

    op->set_ctc_merge_repeated(false);
    EXPECT_EQ(op->get_ctc_merge_repeated(), false);

    op->set_ctc_merge_repeated(true);
    EXPECT_EQ(op->get_ctc_merge_repeated(), true);

    op->validate_and_infer_types();

    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_shape);
}

TEST(type_prop, ctc_greedy_decoder_static_shapes) {
    PartialShape logits_shape{100, 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    Shape out_shape{3, 100, 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_interval_symboled_dims_all) {
    PartialShape data_shape{{1, 100}, {2, 6}, {600, 1200}};
    PartialShape seq_mask_shape{{10, 1000}, {4, 8}};
    PartialShape expected_shape{{4, 6}, {10, 100}, 1, 1};

    auto symbols = set_shape_symbols(data_shape);
    set_shape_symbols(seq_mask_shape);

    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[1], symbols[0], nullptr, nullptr));
}

TEST(type_prop, ctc_greedy_decoder_interval_symboled_dims_data) {
    PartialShape data_shape{{1, 100}, {2, 6}, {600, 1200}};
    PartialShape seq_mask_shape{{10, 1000}, {4, 8}};
    PartialShape expected_shape{{4, 6}, {10, 100}, 1, 1};

    auto symbols = set_shape_symbols(data_shape);

    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[1], symbols[0], nullptr, nullptr));
}

TEST(type_prop, ctc_greedy_decoder_interval_symboled_dims_mask) {
    PartialShape data_shape{{1, 100}, {2, 6}, {600, 1200}};
    PartialShape seq_mask_shape{{10, 1000}, {4, 8}};
    PartialShape expected_shape{{4, 6}, {10, 100}, 1, 1};

    auto symbols = set_shape_symbols(seq_mask_shape);

    auto data = make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);

    const auto& out_shape = op->get_output_partial_shape(0);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(out_shape, expected_shape);
    EXPECT_THAT(get_shape_symbols(out_shape), ElementsAre(symbols[1], symbols[0], nullptr, nullptr));
}

TEST(type_prop, ctc_greedy_decoder_output_static_shape1) {
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    Shape out_shape{3, 100, 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_output_static_shape2) {
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, Dimension::dynamic()};
    Shape out_shape{3, 100, 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
    EXPECT_EQ(op->get_element_type(), element::f32);
    EXPECT_EQ(op->get_shape(), out_shape);
}

TEST(type_prop, ctc_greedy_decoder_dynamic_shapes) {
    PartialShape logits_shape{Dimension::dynamic(), Dimension::dynamic(), 1200};
    PartialShape seq_mask_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape out_shape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_dynamic_ranks1) {
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_mask_shape{100, Dimension::dynamic()};
    PartialShape out_shape{Dimension::dynamic(), 100, 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_dynamic_ranks2) {
    PartialShape logits_shape = PartialShape::dynamic();
    PartialShape seq_mask_shape = PartialShape::dynamic();
    PartialShape out_shape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);
    auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
    EXPECT_EQ(op->get_element_type(), element::f32);
    ASSERT_TRUE(op->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, ctc_greedy_decoder_incorrect_rank) {
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200, 5};
    PartialShape seq_mask_shape{100, 3};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);

    try {
        auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The rank of logits tensor must be equal to 3."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_incorrect_rank2) {
    PartialShape logits_shape{Dimension::dynamic(), 3, 1200};
    PartialShape seq_mask_shape{100, 3, 2};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);

    try {
        auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The rank of sequence mask tensor must be equal to 2."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_mismatched_dim1) {
    PartialShape logits_shape{100, 4, 1200};
    PartialShape seq_mask_shape{100, 3};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);

    try {
        auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The second dimensions of input tensors must match."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_greedy_decoder_mismatched_dim2) {
    PartialShape logits_shape{101, 3, 1200};
    PartialShape seq_mask_shape{100, 3};
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, logits_shape);
    auto seq_mask = make_shared<ov::op::v0::Parameter>(element::f32, seq_mask_shape);

    try {
        auto op = make_shared<op::v0::CTCGreedyDecoder>(data, seq_mask, false);
        // Should have thrown, so fail if it didn't
        FAIL() << "Incorrect indices rank";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("The first dimensions of input tensors must match."));
    } catch (...) {
        FAIL() << "Rank check failed for unexpected reason";
    }
}
