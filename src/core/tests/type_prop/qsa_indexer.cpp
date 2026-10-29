// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "visitors/visitors.hpp"

namespace {

using namespace ov;
using ov::op::v0::Constant;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

OutputVector make_prepare_inputs(const PartialShape& raw_shape = PartialShape{2, 3, 128},
                                 bool disable_optional_inputs = false) {
    const auto empty = Constant::create(element::f32, Shape{0}, std::vector<float>{});
    const auto raw_keys = std::make_shared<Parameter>(element::f32, raw_shape);
    const auto past_blocks = std::make_shared<Parameter>(element::f32, PartialShape{2, 2, 128});
    const auto key_length = Constant::create(element::i32, Shape{2}, std::vector<int32_t>{10, 6});
    const auto rope_cos = disable_optional_inputs
                              ? Output<Node>(empty)
                              : Output<Node>(std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 16}));
    const auto rope_sin = disable_optional_inputs
                              ? Output<Node>(empty)
                              : Output<Node>(std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 16}));
    const auto norm_weight = disable_optional_inputs
                                 ? Output<Node>(empty)
                                 : Output<Node>(std::make_shared<Parameter>(element::f32, PartialShape{128}));
    const auto incomplete_keys = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 128});
    const auto incomplete_rope = disable_optional_inputs
                                     ? Output<Node>(empty)
                                     : Output<Node>(std::make_shared<Parameter>(element::f32, PartialShape{2, 32}));
    return {raw_keys, past_blocks, key_length, rope_cos, rope_sin, norm_weight, incomplete_keys, incomplete_rope};
}

OutputVector make_sparse_indexer_inputs(const PartialShape& query_shape = PartialShape{2, 4, 3, 128},
                                        bool with_weights = false) {
    const auto query = std::make_shared<Parameter>(element::f32, query_shape);
    const auto blocks = std::make_shared<Parameter>(element::f32, PartialShape{2, 8, 128});
    const auto key_length = std::make_shared<Parameter>(element::i32, PartialShape{2});
    if (!with_weights) {
        return {query, blocks, key_length};
    }
    return {query, blocks, key_length, std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 4})};
}

OutputVector make_qsa_indexer_inputs(const PartialShape& query_shape = PartialShape{2, 4, 3, 128}) {
    return {std::make_shared<Parameter>(element::f32, query_shape),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 128}),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 16}),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 16}),
            std::make_shared<Parameter>(element::i32, PartialShape{2}),
            std::make_shared<Parameter>(element::f32, PartialShape{128}),
            std::make_shared<Parameter>(element::f32, PartialShape{128}),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 4, 128}),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 128}),
            std::make_shared<Parameter>(element::f32, PartialShape{2, 32})};
}

TEST(QSATypeProp, PrepareInfersBlocksAndPreservesStates) {
    const auto op = std::make_shared<op::v17::QSAPrepare>(make_prepare_inputs(), 4, 1e-6f);

    EXPECT_EQ(op->get_output_size(), 3);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 2, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 32}));
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_key_block_size(), 4);
    EXPECT_FLOAT_EQ(op->get_epsilon(), 1e-6f);
}

TEST(QSATypeProp, PrepareAllowsAllOptionalFeaturesToBeDisabledTogether) {
    const auto op = std::make_shared<op::v17::QSAPrepare>(make_prepare_inputs(PartialShape{2, 3, 128}, true), 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 2, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 0}));
}

TEST(QSATypeProp, PrepareSupportsDynamicBatchAndSequence) {
    auto inputs = make_prepare_inputs(PartialShape{-1, -1, 128});
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 128});
    inputs[2] = std::make_shared<Parameter>(element::i64, PartialShape{-1});
    inputs[3] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 16});
    inputs[4] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 16});
    inputs[5] = std::make_shared<Parameter>(element::f32, PartialShape{128});
    inputs[6] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 3, 128});
    inputs[7] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 32});
    const auto op = std::make_shared<op::v17::QSAPrepare>(inputs, 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1, 128}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{-1, 32}));
}

TEST(QSATypeProp, PrepareRejectsIncompleteRoPEOptionalInputs) {
    auto inputs = make_prepare_inputs(PartialShape{2, 3, 128}, true);
    inputs[4] = std::make_shared<Parameter>(element::f32, PartialShape{2, 3, 16});
    EXPECT_THROW(std::make_shared<op::v17::QSAPrepare>(inputs, 4), NodeValidationFailure);
}

TEST(QSATypeProp, PrepareRejectsInvalidBlockSizeAndPortCount) {
    EXPECT_THROW(std::make_shared<op::v17::QSAPrepare>(make_prepare_inputs(), 1), NodeValidationFailure);
    EXPECT_THROW(std::make_shared<op::v17::QSAPrepare>(make_prepare_inputs(), 4, 0.0f), NodeValidationFailure);
    auto inputs = make_prepare_inputs();
    inputs.pop_back();
    EXPECT_THROW(std::make_shared<op::v17::QSAPrepare>(inputs, 4), NodeValidationFailure);
}

TEST(QSATypeProp, SparseIndexerInfersSelectionShapesAndSupportsOptionalWeights) {
    const auto op =
        std::make_shared<op::v17::SparseAttentionIndexer>(make_sparse_indexer_inputs(PartialShape{2, 4, 3, 128}, true),
                                                          4,
                                                          32,
                                                          1);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, 3}));
}

TEST(QSATypeProp, SparseIndexerSupportsDynamicShapes) {
    auto inputs = make_sparse_indexer_inputs(PartialShape{-1, 4, -1, 128});
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 128});
    inputs[2] = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto op = std::make_shared<op::v17::SparseAttentionIndexer>(inputs, 4, 32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 8}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1, 1, -1}));
}

TEST(QSATypeProp, SparseIndexerRejectsIncompatibleAttributesAndShapes) {
    EXPECT_THROW(std::make_shared<op::v17::SparseAttentionIndexer>(make_sparse_indexer_inputs(), 4, 31),
                 NodeValidationFailure);
    EXPECT_THROW(std::make_shared<op::v17::SparseAttentionIndexer>(make_sparse_indexer_inputs(), 4, 32, 0),
                 NodeValidationFailure);
    auto inputs = make_sparse_indexer_inputs();
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{2, 8, 64});
    EXPECT_THROW(std::make_shared<op::v17::SparseAttentionIndexer>(inputs, 4, 32), NodeValidationFailure);
}

TEST(QSATypeProp, QSAIndexerInfersSelectionAndStateShapes) {
    const auto op =
        std::make_shared<op::internal::QSAIndexer>(make_qsa_indexer_inputs(),
                                                   op::internal::QSAIndexer::Config{4, 16, 1, 1e-6f, 1e-6f, true});
    EXPECT_EQ(op->get_output_size(), 5);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 3, 4}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, 3}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{2, 4, 128}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{2, 3, 128}));
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{2, 32}));
}

TEST(QSATypeProp, QSAIndexerSupportsDynamicBatchAndSequence) {
    auto inputs = make_qsa_indexer_inputs(PartialShape{-1, 4, -1, 128});
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 128});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 16});
    inputs[3] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 16});
    inputs[4] = std::make_shared<Parameter>(element::i32, PartialShape{1});
    inputs[7] = std::make_shared<Parameter>(element::f32, PartialShape{-1, -1, 128});
    inputs[8] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 3, 128});
    inputs[9] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 32});
    const auto op =
        std::make_shared<op::internal::QSAIndexer>(inputs,
                                                   op::internal::QSAIndexer::Config{4, 16, 1, 1e-6f, 1e-6f, true});
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 4}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{-1, -1, 128}));
    EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{-1, 3, 128}));
}

TEST(QSATypeProp, QSAIndexerInfersDisabledRoPEStateShape) {
    auto inputs = make_qsa_indexer_inputs();
    const auto empty = Constant::create(element::f32, Shape{0}, std::vector<float>{});
    inputs[2] = empty;
    inputs[3] = empty;
    inputs[9] = empty;
    const auto op =
        std::make_shared<op::internal::QSAIndexer>(inputs,
                                                   op::internal::QSAIndexer::Config{4, 16, 1, 1e-6f, 1e-6f, true});
    EXPECT_EQ(op->get_output_partial_shape(4), (PartialShape{2, 0}));
}

TEST(QSATypeProp, QSAIndexerRejectsNonCausalConfigurationAndInvalidPorts) {
    auto config = op::internal::QSAIndexer::Config{4, 16, 1, 1e-6f, 1e-6f, false};
    EXPECT_THROW(std::make_shared<op::internal::QSAIndexer>(make_qsa_indexer_inputs(), config), NodeValidationFailure);
    auto inputs = make_qsa_indexer_inputs();
    inputs.pop_back();
    EXPECT_THROW(
        std::make_shared<op::internal::QSAIndexer>(inputs,
                                                   op::internal::QSAIndexer::Config{4, 16, 1, 1e-6f, 1e-6f, true}),
        NodeValidationFailure);
    EXPECT_THROW(std::make_shared<op::internal::QSAIndexer>(
                     make_qsa_indexer_inputs(),
                     op::internal::QSAIndexer::Config{4, 16, 1, 0.0f, 1e-6f, true}),
                 NodeValidationFailure);
}

TEST(QSATypeProp, QSAIndexerReportsKernelSupportWithoutThrowing) {
    using Config = op::internal::QSAIndexer::Config;
    const auto supported_config = Config{4, 512, 1, 1e-5f, 1e-6f, true};
    const auto supported_query = PartialShape{-1, 4, -1, 128};
    std::string reason;
    EXPECT_TRUE(op::internal::QSAIndexer::is_supported(supported_config, supported_query, reason));
    EXPECT_TRUE(reason.empty());

    const std::vector<std::pair<Config, PartialShape>> unsupported_cases{
        {Config{4, 16, 1, 1e-6f, 1e-6f, false}, supported_query},
        {Config{8, 16, 1, 1e-6f, 1e-6f, true}, supported_query},
        {Config{4, 16, 2, 1e-6f, 1e-6f, true}, supported_query},
        {Config{4, 516, 1, 1e-6f, 1e-6f, true}, supported_query},
        {Config{4, 18, 1, 1e-6f, 1e-6f, true}, supported_query},
        {Config{4, 16, 1, 5e-7f, 1e-6f, true}, supported_query},
        {Config{4, 16, 1, 1e-6f, 2e-5f, true}, supported_query},
        {Config{4, 16, 1, 1e-6f, 1e-6f, true}, PartialShape::dynamic()},
        {Config{4, 16, 1, 1e-6f, 1e-6f, true}, PartialShape{2, 8, 3, 128}},
        {Config{4, 16, 1, 1e-6f, 1e-6f, true}, PartialShape{2, -1, 3, 128}},
        {Config{4, 16, 1, 1e-6f, 1e-6f, true}, PartialShape{2, 4, 3, 64}},
        {Config{4, 16, 1, 1e-6f, 1e-6f, true}, PartialShape{2, 4, 3, -1}},
    };
    for (const auto& [config, query_shape] : unsupported_cases) {
        EXPECT_FALSE(op::internal::QSAIndexer::is_supported(config, query_shape, reason)) << query_shape;
        EXPECT_FALSE(reason.empty()) << query_shape;
    }
}

TEST(QSATypeProp, OperationAttributesRoundTripThroughVisitors) {
    NodeBuilder::opset().insert<op::v17::QSAPrepare>();
    NodeBuilder::opset().insert<op::v17::SparseAttentionIndexer>();
    NodeBuilder::opset().insert<op::internal::QSAIndexer>();

    const auto prepare_inputs = make_prepare_inputs();
    const auto prepare = std::make_shared<op::v17::QSAPrepare>(prepare_inputs, 4, 2e-6f);
    auto cloned_prepare = ov::as_type_ptr<op::v17::QSAPrepare>(NodeBuilder(prepare, prepare_inputs).create());
    ASSERT_NE(cloned_prepare, nullptr);
    EXPECT_EQ(cloned_prepare->get_key_block_size(), 4);
    EXPECT_FLOAT_EQ(cloned_prepare->get_epsilon(), 2e-6f);
    auto cloned_prepare_with_new_inputs =
        ov::as_type_ptr<op::v17::QSAPrepare>(prepare->clone_with_new_inputs(prepare->input_values()));
    ASSERT_NE(cloned_prepare_with_new_inputs, nullptr);
    EXPECT_EQ(cloned_prepare_with_new_inputs->get_key_block_size(), 4);
    EXPECT_FLOAT_EQ(cloned_prepare_with_new_inputs->get_epsilon(), 2e-6f);

    const auto indexer_inputs = make_sparse_indexer_inputs();
    const auto indexer = std::make_shared<op::v17::SparseAttentionIndexer>(indexer_inputs, 4, 32, 1);
    auto cloned_indexer =
        ov::as_type_ptr<op::v17::SparseAttentionIndexer>(NodeBuilder(indexer, indexer_inputs).create());
    ASSERT_NE(cloned_indexer, nullptr);
    EXPECT_EQ(cloned_indexer->get_key_block_size(), 4);
    EXPECT_EQ(cloned_indexer->get_token_budget(), 32);
    EXPECT_EQ(cloned_indexer->get_num_selection_heads(), 1);
    auto cloned_indexer_with_new_inputs =
        ov::as_type_ptr<op::v17::SparseAttentionIndexer>(indexer->clone_with_new_inputs(indexer->input_values()));
    ASSERT_NE(cloned_indexer_with_new_inputs, nullptr);
    EXPECT_EQ(cloned_indexer_with_new_inputs->get_token_budget(), 32);

    const auto fused_inputs = make_qsa_indexer_inputs();
    const auto config = op::internal::QSAIndexer::Config{4, 16, 1, 2e-6f, 3e-6f, true};
    const auto fused = std::make_shared<op::internal::QSAIndexer>(fused_inputs, config);
    auto cloned_fused = ov::as_type_ptr<op::internal::QSAIndexer>(NodeBuilder(fused, fused_inputs).create());
    ASSERT_NE(cloned_fused, nullptr);
    EXPECT_EQ(cloned_fused->get_config().key_block_size, 4);
    EXPECT_EQ(cloned_fused->get_config().token_budget, 16);
    EXPECT_FLOAT_EQ(cloned_fused->get_config().q_epsilon, 2e-6f);
    EXPECT_FLOAT_EQ(cloned_fused->get_config().k_epsilon, 3e-6f);
    EXPECT_TRUE(cloned_fused->get_config().is_causal);
    auto cloned_fused_with_new_inputs =
        ov::as_type_ptr<op::internal::QSAIndexer>(fused->clone_with_new_inputs(fused->input_values()));
    ASSERT_NE(cloned_fused_with_new_inputs, nullptr);
    EXPECT_EQ(cloned_fused_with_new_inputs->get_config().token_budget, 16);
}

}  // namespace
