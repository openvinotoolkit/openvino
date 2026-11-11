// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <tuple>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::testing {
namespace {

using op::internal::PagedQSAIndexer;
using op::v0::Parameter;

// Consistent configuration: key block size r = 4, K = 3, one selection head, D_idx = 4 with 2 rotary dims.
PagedQSAIndexer::Config make_paged_qsa_indexer_config() {
    PagedQSAIndexer::Config config;
    config.key_block_size = 4;
    config.block_topk = 3;
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = 2;
    config.indexer_head_dim = 4;
    return config;
}

// Six tokens, eight physical pages of page_size slots, query/key heads of width D_idx = 4.
OutputVector make_paged_qsa_indexer_inputs(int64_t query_heads = 2, int64_t key_heads = 1, int64_t page_size = 8) {
    constexpr int64_t head_dim = 4;
    constexpr int64_t rotary_dim = 2;
    constexpr int64_t num_pages = 8;
    const int64_t summaries_per_page = page_size / 4;  // page_size / key_block_size
    return {
        std::make_shared<Parameter>(element::f32, PartialShape{6, query_heads * head_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{6, key_heads * head_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{6, rotary_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{6, rotary_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{head_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{head_dim}),
        std::make_shared<Parameter>(element::i32, PartialShape{2}),
        std::make_shared<Parameter>(element::i32, PartialShape{3}),
        std::make_shared<Parameter>(element::i32, PartialShape{4}),
        std::make_shared<Parameter>(element::i32, PartialShape{3}),
        std::make_shared<Parameter>(element::f32, PartialShape{num_pages, key_heads, page_size, head_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{num_pages, summaries_per_page, 2 * rotary_dim}),
        std::make_shared<Parameter>(element::f32, PartialShape{num_pages, key_heads, summaries_per_page, head_dim})};
}

}  // namespace

TEST(type_prop, paged_qsa_indexer_output_shape_and_type) {
    const auto op = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(), make_paged_qsa_indexer_config());

    // Both outputs are i32 selections; the paged caches are updated in place and are not outputs.
    ASSERT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    // Six tokens, one selection head, block_topk = 3.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 1, 3}));
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{6, 1}));
    EXPECT_EQ(op->get_index_query_heads(), 2u);
    EXPECT_EQ(op->get_index_key_heads(), 1u);
}

TEST(type_prop, paged_qsa_indexer_per_key_head_selection) {
    auto config = make_paged_qsa_indexer_config();
    // num_selection_heads may equal H_ik, giving one selection row per key head.
    config.num_selection_heads = 2;
    const auto op = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(2, 2), config);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 2, 3}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{6, 2}));
    EXPECT_EQ(op->get_index_key_heads(), 2u);
}

TEST(type_prop, paged_qsa_indexer_dynamic_tokens_and_pages) {
    auto inputs = make_paged_qsa_indexer_inputs();
    const auto dyn = Dimension::dynamic();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, dyn});
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, dyn});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, 2});
    inputs[3] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, 2});
    inputs[10] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, dyn, dyn, dyn});
    inputs[11] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, dyn, dyn});
    inputs[12] = std::make_shared<Parameter>(element::f32, PartialShape{dyn, dyn, dyn, dyn});
    const auto op = std::make_shared<PagedQSAIndexer>(inputs, make_paged_qsa_indexer_config());

    // The token count is unknown, while the selection width and the selection head count stay static.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{dyn, 1, 3}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{dyn, 1}));
}

TEST(type_prop, paged_qsa_indexer_rejects_invalid_config) {
    const auto expect_invalid = [](const PagedQSAIndexer::Config& config) {
        EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(), config),
                     NodeValidationFailure);
    };

    // Each attribute constraint enforced by validate_config must fail at construction time.
    auto config = make_paged_qsa_indexer_config();
    config.key_block_size = 0;
    expect_invalid(config);

    config = make_paged_qsa_indexer_config();
    config.block_topk = 0;
    expect_invalid(config);

    config = make_paged_qsa_indexer_config();
    config.num_selection_heads = 0;
    expect_invalid(config);

    config = make_paged_qsa_indexer_config();
    config.indexer_rotary_dim = 3;  // must be even
    expect_invalid(config);

    config = make_paged_qsa_indexer_config();
    config.indexer_rotary_dim = 6;  // must not exceed indexer_head_dim
    expect_invalid(config);

    config = make_paged_qsa_indexer_config();
    config.scale = -1.0f;
    expect_invalid(config);
}

TEST(type_prop, paged_qsa_indexer_rejects_incompatible_shapes) {
    const auto config = make_paged_qsa_indexer_config();
    auto inputs = make_paged_qsa_indexer_inputs();

    // The query width must be a whole number of indexer_head_dim heads.
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 6});
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, config), NodeValidationFailure);

    inputs = make_paged_qsa_indexer_inputs();
    // Raw keys must describe the same token stream as the index queries.
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{5, 4});
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, config), NodeValidationFailure);

    inputs = make_paged_qsa_indexer_inputs();
    // RoPE rows carry indexer_rotary_dim coefficients per token.
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{6, 3});
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, config), NodeValidationFailure);

    // page_size 6 is not a multiple of key_block_size 4.
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(2, 1, 6), config),
                 NodeValidationFailure);

    // H_iq = 3 is not a multiple of H_ik = 2.
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(3, 2), config),
                 NodeValidationFailure);

    inputs = make_paged_qsa_indexer_inputs();
    // The three paged caches must share the physical page axis.
    inputs[12] = std::make_shared<Parameter>(element::f32, PartialShape{7, 1, 2, 4});
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, config), NodeValidationFailure);

    // num_selection_heads must be 1 or H_ik; here H_ik = 1.
    auto bad_selection = make_paged_qsa_indexer_config();
    bad_selection.num_selection_heads = 2;
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(make_paged_qsa_indexer_inputs(), bad_selection),
                 NodeValidationFailure);
}

TEST(type_prop, paged_qsa_indexer_rejects_invalid_types_and_arity) {
    auto inputs = make_paged_qsa_indexer_inputs();
    inputs.pop_back();
    // All 13 inputs are required.
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, make_paged_qsa_indexer_config()),
                 NodeValidationFailure);

    inputs = make_paged_qsa_indexer_inputs();
    inputs[6] = std::make_shared<Parameter>(element::i64, PartialShape{2});
    // The kernels read every integer input as i32, so i64 index tables are rejected.
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, make_paged_qsa_indexer_config()),
                 NodeValidationFailure);

    inputs = make_paged_qsa_indexer_inputs();
    inputs[4] = std::make_shared<Parameter>(element::f16, PartialShape{4});
    // Floating-point inputs must share one element type.
    EXPECT_THROW(std::ignore = std::make_shared<PagedQSAIndexer>(inputs, make_paged_qsa_indexer_config()),
                 NodeValidationFailure);
}

TEST(type_prop, paged_qsa_indexer_effective_scale_defaults_to_inverse_sqrt_head_dim) {
    auto config = make_paged_qsa_indexer_config();
    // scale = 0 selects 1 / sqrt(indexer_head_dim) = 1 / 2 for D_idx = 4.
    EXPECT_FLOAT_EQ(PagedQSAIndexer::effective_scale(config), 0.5f);
    config.scale = 0.25f;
    EXPECT_FLOAT_EQ(PagedQSAIndexer::effective_scale(config), 0.25f);
}

}  // namespace ov::testing
