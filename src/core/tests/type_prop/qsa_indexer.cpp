// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter;

class TypePropQSAIndexerTest : public TypePropOpTest<op::internal::QSAIndexer> {
protected:
    using Config = op::internal::QSAIndexer::Config;
    using InputIdx = op::internal::QSAIndexer::InputIdx;
    using OutputIdx = op::internal::QSAIndexer::OutputIdx;

    static std::shared_ptr<Parameter> param(element::Type et, const PartialShape& shape) {
        return std::make_shared<Parameter>(et, shape);
    }

    /// Inputs with B = 2, L = 3, H_iq = 4, H_ik = 1, D = 128, a rotary dimension of 16, r = 4 and 5 cached blocks.
    static OutputVector make_inputs() {
        return {param(element::f32, PartialShape{2, 3, 4, 128}),
                param(element::f32, PartialShape{2, 3, 1, 128}),
                param(element::f32, PartialShape{2, 3, 16}),
                param(element::f32, PartialShape{2, 3, 16}),
                param(element::i32, PartialShape{2}),
                param(element::f32, PartialShape{128}),
                param(element::f32, PartialShape{128}),
                param(element::f32, PartialShape{2, 1, 5, 128}),
                param(element::f32, PartialShape{2, 1, 3, 128}),
                param(element::f32, PartialShape{2, 32})};
    }

    static Config config() {
        Config config;
        config.key_block_size = 4;
        config.block_topk = 4;
        config.indexer_rotary_dim = 16;
        config.indexer_head_dim = 128;
        return config;
    }
};

TEST_F(TypePropQSAIndexerTest, infers_selection_and_state_shapes) {
    const auto op = make_op(make_inputs(), config());
    EXPECT_EQ(op->get_output_size(), OutputIdx::COUNT);
    EXPECT_EQ(op->get_output_element_type(OutputIdx::SELECTED_BLOCK_INDICES), element::i32);
    EXPECT_EQ(op->get_output_element_type(OutputIdx::SELECTED_BLOCK_COUNT), element::i32);
    EXPECT_EQ(op->get_output_element_type(OutputIdx::BLOCK_INDEX_KEYS_CACHE), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::SELECTED_BLOCK_INDICES), (PartialShape{2, 1, 3, 4}));
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::SELECTED_BLOCK_COUNT), (PartialShape{2, 1, 3}));
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::BLOCK_INDEX_KEYS_CACHE), (PartialShape{2, 1, 5, 128}));
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::PAST_INCOMPLETE_RAW_K), (PartialShape{2, 1, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::PAST_INCOMPLETE_ROPE), (PartialShape{2, 32}));
    EXPECT_FLOAT_EQ(op::internal::QSAIndexer::effective_scale(config()), 1.0f / std::sqrt(128.0f));
}

TEST_F(TypePropQSAIndexerTest, dynamic_batch_and_sequence) {
    auto inputs = make_inputs();
    inputs[InputIdx::INDEX_QUERY] = param(element::f32, PartialShape{-1, -1, 4, 128});
    inputs[InputIdx::RAW_INDEX_KEYS] = param(element::f32, PartialShape{-1, -1, 1, 128});
    inputs[InputIdx::ROPE_COS] = param(element::f32, PartialShape{-1, -1, 16});
    inputs[InputIdx::ROPE_SIN] = param(element::f32, PartialShape{-1, -1, 16});
    inputs[InputIdx::KEY_LENGTH] = param(element::i32, PartialShape{-1});
    inputs[InputIdx::BLOCK_INDEX_KEYS_CACHE] = param(element::f32, PartialShape{-1, 1, -1, 128});
    inputs[InputIdx::PAST_INCOMPLETE_RAW_K] = param(element::f32, PartialShape{-1, 1, 3, 128});
    inputs[InputIdx::PAST_INCOMPLETE_ROPE] = param(element::f32, PartialShape{-1, 32});
    const auto op = make_op(inputs, config());
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::SELECTED_BLOCK_INDICES), (PartialShape{-1, 1, -1, 4}));
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::BLOCK_INDEX_KEYS_CACHE), (PartialShape{-1, 1, -1, 128}));
}

TEST_F(TypePropQSAIndexerTest, selection_per_key_head) {
    auto inputs = make_inputs();
    inputs[InputIdx::RAW_INDEX_KEYS] = param(element::f32, PartialShape{2, 3, 2, 128});
    inputs[InputIdx::BLOCK_INDEX_KEYS_CACHE] = param(element::f32, PartialShape{2, 2, 5, 128});
    inputs[InputIdx::PAST_INCOMPLETE_RAW_K] = param(element::f32, PartialShape{2, 2, 3, 128});
    auto per_head = config();
    per_head.num_selection_heads = 2;
    const auto op = make_op(inputs, per_head);
    EXPECT_EQ(op->get_output_partial_shape(OutputIdx::SELECTED_BLOCK_INDICES), (PartialShape{2, 2, 3, 4}));

    per_head.num_selection_heads = 3;
    OV_EXPECT_THROW(std::ignore = make_op(inputs, per_head),
                    NodeValidationFailure,
                    testing::HasSubstr("num_selection_heads must be 1 or H_ik"));
}

TEST_F(TypePropQSAIndexerTest, incompatible_shapes) {
    const std::vector<std::pair<size_t, PartialShape>> cases{
        {InputIdx::INDEX_QUERY, PartialShape{2, 3, 4, 64}},
        {InputIdx::RAW_INDEX_KEYS, PartialShape{2, 3, 128}},
        {InputIdx::ROPE_COS, PartialShape{2, 3, 8}},
        {InputIdx::KEY_LENGTH, PartialShape{3}},
        {InputIdx::Q_NORM_WEIGHT, PartialShape{64}},
        {InputIdx::BLOCK_INDEX_KEYS_CACHE, PartialShape{2, 5, 128}},
        {InputIdx::PAST_INCOMPLETE_RAW_K, PartialShape{2, 1, 2, 128}},
        {InputIdx::PAST_INCOMPLETE_ROPE, PartialShape{2, 16}},
    };
    for (const auto& [idx, shape] : cases) {
        auto inputs = make_inputs();
        inputs[idx] = param(inputs[idx].get_element_type(), shape);
        EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure) << "input " << idx;
    }
}

TEST_F(TypePropQSAIndexerTest, incompatible_element_types) {
    auto inputs = make_inputs();
    inputs[InputIdx::K_NORM_WEIGHT] = param(element::f16, PartialShape{128});
    OV_EXPECT_THROW(std::ignore = make_op(inputs, config()),
                    NodeValidationFailure,
                    testing::HasSubstr("share one element type"));

    inputs = make_inputs();
    inputs[InputIdx::KEY_LENGTH] = param(element::i64, PartialShape{2});
    OV_EXPECT_THROW(std::ignore = make_op(inputs, config()),
                    NodeValidationFailure,
                    testing::HasSubstr("i32 key_length"));
}

TEST_F(TypePropQSAIndexerTest, invalid_configuration) {
    const std::vector<std::pair<std::string, Config>> cases{
        {"block_topk",
         [] {
             auto c = config();
             c.block_topk = 0;
             return c;
         }()},
        {"odd rotary",
         [] {
             auto c = config();
             c.indexer_rotary_dim = 15;
             return c;
         }()},
        {"rotary above head size",
         [] {
             auto c = config();
             c.indexer_rotary_dim = 130;
             return c;
         }()},
        {"negative scale",
         [] {
             auto c = config();
             c.scale = -1.0f;
             return c;
         }()},
    };
    for (const auto& [name, op_config] : cases) {
        std::string reason;
        EXPECT_FALSE(op::internal::QSAIndexer::validate_config(op_config, reason)) << name;
        EXPECT_FALSE(reason.empty()) << name;
        OV_EXPECT_THROW(std::ignore = make_op(make_inputs(), op_config),
                        NodeValidationFailure,
                        testing::HasSubstr("configuration is invalid"));
    }
}

TEST_F(TypePropQSAIndexerTest, incorrect_number_of_inputs) {
    auto inputs = make_inputs();
    inputs.pop_back();
    EXPECT_THROW(std::ignore = make_op(inputs, config()), NodeValidationFailure);
}
}  // namespace ov::test
