// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/type_prop.hpp"

namespace ov::test {
namespace {
using op::internal::PagedQSAIndexer;
using op::v0::Parameter;
using PagedQSAIndexerInputIdx = PagedQSAIndexer::InputIdx;

OutputVector make_inputs(element::Type type = element::f32) {
    const std::vector<PartialShape> shapes{{6, 64},
                                           {6, 32},
                                           {6, 8},
                                           {6, 8},
                                           {16},
                                           {16},
                                           {2},
                                           {3},
                                           {10},
                                           {3},
                                           {5, 2, 16, 16},
                                           {5, 4, 16},
                                           {5, 2, 4, 16}};
    OutputVector inputs;
    for (size_t idx = 0; idx < shapes.size(); ++idx) {
        const bool metadata =
            idx >= PagedQSAIndexerInputIdx::PAST_LENS && idx <= PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS;
        inputs.push_back(std::make_shared<Parameter>(metadata ? element::i32 : type, shapes[idx]));
    }
    return inputs;
}

PagedQSAIndexer::Config config() {
    return {4, 2, 1, 8, 16, 1e-6f, 0.0f};
}
}  // namespace

class TypePropPagedQSAIndexerTest : public TypePropOpTest<PagedQSAIndexer> {};

TEST_F(TypePropPagedQSAIndexerTest, static_shapes_and_default_scale) {
    const auto op = make_op(make_inputs(), config());
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 1, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{6, 1}));
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_index_query_heads(), 4);
    EXPECT_EQ(op->get_index_key_heads(), 2);
    EXPECT_FLOAT_EQ(op->get_config().scale, 0.25f);
    EXPECT_TRUE(op->get_config().query_norm);
    EXPECT_TRUE(op->get_config().query_rope);
    const auto clone = ov::as_type_ptr<PagedQSAIndexer>(op->clone_with_new_inputs(make_inputs()));
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_FLOAT_EQ(clone->get_config().scale, op->get_config().scale);
}

TEST_F(TypePropPagedQSAIndexerTest, optional_query_processing_and_clone) {
    for (const bool query_norm : {false, true}) {
        for (const bool query_rope : {false, true}) {
            auto c = config();
            c.query_norm = query_norm;
            c.query_rope = query_rope;
            const auto op = make_op(make_inputs(), c);
            const auto clone = ov::as_type_ptr<PagedQSAIndexer>(op->clone_with_new_inputs(make_inputs()));
            ASSERT_NE(clone, nullptr);
            EXPECT_EQ(op->get_config().query_norm, query_norm);
            EXPECT_EQ(op->get_config().query_rope, query_rope);
            EXPECT_EQ(clone->get_config().query_norm, query_norm);
            EXPECT_EQ(clone->get_config().query_rope, query_rope);
            EXPECT_EQ(clone->get_output_partial_shape(0), op->get_output_partial_shape(0));
        }
    }
}

TEST_F(TypePropPagedQSAIndexerTest, floating_point_precisions) {
    for (const auto& type : {element::f32, element::f16, element::bf16}) {
        EXPECT_NO_THROW(std::ignore = make_op(make_inputs(type), config()));
    }
    EXPECT_THROW(std::ignore = make_op(make_inputs(element::f64), config()), NodeValidationFailure);
}

TEST_F(TypePropPagedQSAIndexerTest, dynamic_input_ranks) {
    auto inputs = make_inputs();
    for (auto& input : inputs) {
        input = std::make_shared<Parameter>(input.get_element_type(), PartialShape::dynamic());
    }
    const auto op = make_op(inputs, config());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 1, 2}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1, 1}));
    EXPECT_THROW(std::ignore = op->get_index_query_heads(), ov::AssertFailure);
}

TEST_F(TypePropPagedQSAIndexerTest, selection_per_key_head) {
    auto c = config();
    c.num_selection_heads = 2;
    const auto op = make_op(make_inputs(), c);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 2, 2}));
    c.num_selection_heads = 3;
    EXPECT_THROW(std::ignore = make_op(make_inputs(), c), NodeValidationFailure);
}

TEST_F(TypePropPagedQSAIndexerTest, partially_dynamic_shared_dimensions) {
    auto inputs = make_inputs();
    inputs[PagedQSAIndexerInputIdx::INDEX_QUERY] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 64});
    inputs[PagedQSAIndexerInputIdx::RAW_INDEX_KEYS] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1});
    const auto op = make_op(inputs, config());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 1, 2}));

    auto inconsistent = inputs;
    inconsistent[PagedQSAIndexerInputIdx::ROPE_SIN] = std::make_shared<Parameter>(element::f32, PartialShape{7, 8});
    EXPECT_THROW(std::ignore = make_op(inconsistent, config()), NodeValidationFailure);

    inconsistent = inputs;
    inconsistent[PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] =
        std::make_shared<Parameter>(element::f32, PartialShape{5, 3, 4, 16});
    EXPECT_THROW(std::ignore = make_op(inconsistent, config()), NodeValidationFailure);

    inconsistent = inputs;
    inconsistent[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE] =
        std::make_shared<Parameter>(element::f32, PartialShape{5, 2, -1, 16});
    inconsistent[PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE] =
        std::make_shared<Parameter>(element::f32, PartialShape{5, 2, 3, 16});
    EXPECT_THROW(std::ignore = make_op(inconsistent, config()), NodeValidationFailure);

    inconsistent = inputs;
    inconsistent[PagedQSAIndexerInputIdx::PAST_LENS] = std::make_shared<Parameter>(element::i32, PartialShape{-1});
    inconsistent[PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS] =
        std::make_shared<Parameter>(element::i32, PartialShape{4});
    EXPECT_THROW(std::ignore = make_op(inconsistent, config()), NodeValidationFailure);
}

TEST_F(TypePropPagedQSAIndexerTest, incorrect_ranks) {
    const auto inputs = make_inputs();
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        SCOPED_TRACE(idx);
        auto wrong_rank_inputs = inputs;
        wrong_rank_inputs[idx] =
            std::make_shared<Parameter>(inputs[idx].get_element_type(),
                                        PartialShape::dynamic(inputs[idx].get_partial_shape().size() + 1));
        EXPECT_THROW(std::ignore = make_op(wrong_rank_inputs, config()), NodeValidationFailure);
    }
}

TEST_F(TypePropPagedQSAIndexerTest, incompatible_shapes) {
    const std::vector<std::pair<size_t, PartialShape>> cases{
        {PagedQSAIndexerInputIdx::INDEX_QUERY, {6, 63}},
        {PagedQSAIndexerInputIdx::RAW_INDEX_KEYS, {7, 32}},
        {PagedQSAIndexerInputIdx::RAW_INDEX_KEYS, {6, 48}},
        {PagedQSAIndexerInputIdx::RAW_INDEX_KEYS, {6, 0}},
        {PagedQSAIndexerInputIdx::ROPE_COS, {6, 4}},
        {PagedQSAIndexerInputIdx::ROPE_SIN, {7, 8}},
        {PagedQSAIndexerInputIdx::Q_NORM_WEIGHT, {8}},
        {PagedQSAIndexerInputIdx::K_NORM_WEIGHT, {8}},
        {PagedQSAIndexerInputIdx::SUBSEQUENCE_BEGINS, {2}},
        {PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS, {2}},
        {PagedQSAIndexerInputIdx::INDEX_KEY_CACHE, {5, 2, 15, 16}},
        {PagedQSAIndexerInputIdx::INDEX_KEY_CACHE, {5, 2, 0, 16}},
        {PagedQSAIndexerInputIdx::INDEX_BLOCK_ROPE_CACHE, {6, 4, 16}},
        {PagedQSAIndexerInputIdx::INDEX_BLOCK_ROPE_CACHE, {5, 4, 8}},
        {PagedQSAIndexerInputIdx::BLOCK_INDEX_KEYS_CACHE, {5, 2, 3, 16}},
    };
    for (const auto& [idx, shape] : cases) {
        SCOPED_TRACE(idx);
        auto inputs = make_inputs();
        inputs[idx] = std::make_shared<Parameter>(inputs[idx].get_element_type(), shape);
        EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
    }
}

TEST_F(TypePropPagedQSAIndexerTest, incompatible_types_and_input_count) {
    auto inputs = make_inputs();
    inputs[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE] =
        std::make_shared<Parameter>(element::f16, inputs[PagedQSAIndexerInputIdx::INDEX_KEY_CACHE].get_partial_shape());
    EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
    for (size_t idx = PagedQSAIndexerInputIdx::PAST_LENS; idx <= PagedQSAIndexerInputIdx::BLOCK_INDICES_BEGINS; ++idx) {
        inputs = make_inputs();
        inputs[idx] = std::make_shared<Parameter>(element::i64, inputs[idx].get_partial_shape());
        EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
    }
    inputs = make_inputs();
    inputs.pop_back();
    EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
}

TEST_F(TypePropPagedQSAIndexerTest, invalid_attributes) {
    const auto check_invalid = [this](const char* attribute, const auto& invalidate) {
        SCOPED_TRACE(attribute);
        auto c = config();
        invalidate(c);
        EXPECT_THROW(std::ignore = make_op(make_inputs(), c), NodeValidationFailure);
    };
    check_invalid("key_block_size", [](auto& c) {
        c.key_block_size = 0;
    });
    check_invalid("block_topk", [](auto& c) {
        c.block_topk = 0;
    });
    check_invalid("num_selection_heads", [](auto& c) {
        c.num_selection_heads = 0;
    });
    check_invalid("indexer_head_dim", [](auto& c) {
        c.indexer_head_dim = 0;
    });
    check_invalid("indexer_rotary_dim", [](auto& c) {
        c.indexer_rotary_dim = 7;
    });
    check_invalid("indexer_rotary_dim", [](auto& c) {
        c.indexer_rotary_dim = 32;
    });
    check_invalid("epsilon", [](auto& c) {
        c.epsilon = 0;
    });
    check_invalid("scale", [](auto& c) {
        c.scale = std::numeric_limits<float>::quiet_NaN();
    });
}

}  // namespace ov::test
