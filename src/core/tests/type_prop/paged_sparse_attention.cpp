// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"
#include "paged_sparse_attention_shape_inference.hpp"

namespace ov::testing {
namespace {

using op::internal::PagedSparseAttention;
using op::v0::Parameter;

OutputVector make_valid_inputs(const element::Type& query_type = element::f32,
                               const element::Type& key_cache_type = element::f16,
                               const element::Type& value_cache_type = element::bf16) {
    return {std::make_shared<Parameter>(query_type, PartialShape{6, 64}),
            std::make_shared<Parameter>(query_type, PartialShape{6, 32}),
            std::make_shared<Parameter>(query_type, PartialShape{6, 16}),
            std::make_shared<Parameter>(key_cache_type, PartialShape{8, 2, 32, 16}),
            std::make_shared<Parameter>(value_cache_type, PartialShape{8, 2, 32, 8}),
            std::make_shared<Parameter>(element::i32, PartialShape{3}),
            std::make_shared<Parameter>(element::i32, PartialShape{4}),
            std::make_shared<Parameter>(element::i32, PartialShape{12}),
            std::make_shared<Parameter>(element::i32, PartialShape{4}),
            std::make_shared<Parameter>(element::i32, PartialShape{6, 1, 16}),
            std::make_shared<Parameter>(element::i32, PartialShape{6, 1}),
            std::make_shared<Parameter>(query_type, PartialShape{}),
            std::make_shared<Parameter>(element::f16, PartialShape{4})};
}

}  // namespace

TEST(type_prop, paged_sparse_attention_output_shape_and_type) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);

    // Only attention is returned because cache updates are in place.
    EXPECT_EQ(op->get_output_size(), 1);
    // Attention uses the merged query/key/value/scale precision, not the cache precision.
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    // Four query heads times value head size eight give 32 features for each of six tokens.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
}

TEST(type_prop, paged_sparse_attention_dynamic_dimensions) {
    auto inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    inputs[3] = std::make_shared<Parameter>(
        element::f16,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    inputs[4] = std::make_shared<Parameter>(
        element::bf16,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    inputs[9] =
        std::make_shared<Parameter>(element::i32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    inputs[10] = std::make_shared<Parameter>(element::i32, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);

    // Known input ranks fix the output rank, but unknown token/head sizes leave both dimensions dynamic.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, paged_sparse_attention_accepts_quantized_cache_types) {
    const auto key_cache_types =
        {element::dynamic, element::u4, element::i8, element::u8, element::f16, element::f32, element::bf16};
    const auto value_cache_types =
        {element::dynamic, element::u4, element::u8, element::f16, element::f32, element::bf16};
    for (const auto& type : key_cache_types) {
        // Key cache storage precision is independent of query precision and includes signed i8.
        EXPECT_NO_THROW(std::make_shared<PagedSparseAttention>(make_valid_inputs(element::f32, type, element::f16), 4));
    }
    for (const auto& type : value_cache_types) {
        // Value caches accept their specified storage types without forcing the key cache to match.
        EXPECT_NO_THROW(std::make_shared<PagedSparseAttention>(make_valid_inputs(element::f32, element::f16, type), 4));
    }
}

TEST(type_prop, paged_sparse_attention_rejects_invalid_key_block_size) {
    // Zero cannot define a selected block or serve as the page-divisibility divisor.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(make_valid_inputs(), 0), NodeValidationFailure);
    // Negative block lengths cannot describe groups of key tokens.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(make_valid_inputs(), -4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_rejects_incompatible_shapes) {
    auto inputs = make_valid_inputs();
    inputs[3] = std::make_shared<Parameter>(element::f16, PartialShape{8, 2, 30, 16});
    inputs[4] = std::make_shared<Parameter>(element::bf16, PartialShape{8, 2, 30, 8});
    // A 30-token page cannot contain an integral number of four-token selected blocks.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{5, 32});
    // Current query and key projections must describe the same token stream.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 80});
    // Five query heads cannot be grouped evenly over two KV heads.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 64});
    inputs[1] = std::make_shared<Parameter>(element::f16, PartialShape{6, 32});
    // Projection types must match even when their shapes are otherwise valid.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 2, 32});
    // Query heads must be flattened into a rank-two projection.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{6, 2, 8});
    // Value heads must also be flattened rather than exposed as a separate axis.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[10] = std::make_shared<Parameter>(element::i32, PartialShape{6, 2});
    // Selection counts must use the same selection-head axis as the selected indices.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_rejects_invalid_types_and_arity) {
    auto inputs = make_valid_inputs();
    inputs[11] = std::make_shared<Parameter>(element::f16, PartialShape{});
    // Scale shares the query/key/value type rather than having an independent precision.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[7] = std::make_shared<Parameter>(element::i64, PartialShape{12});
    // The page-table requires i32 indices, not merely any integer type.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs.pop_back();
    // All 13 ports are required, including an explicit sinks input when it is empty.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_merges_input_types) {
    for (const auto dynamic_input : {0, 1, 2, 11}) {
        auto inputs = make_valid_inputs();
        inputs[dynamic_input] =
            std::make_shared<Parameter>(element::dynamic, inputs[dynamic_input].get_partial_shape());
        const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
        // Each unknown data type is resolved from the other known query/key/value/scale types.
        EXPECT_EQ(op->get_output_element_type(0), element::f32);
    }

    auto inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::dynamic, PartialShape{6, 64});
    inputs[2] = std::make_shared<Parameter>(element::f16, PartialShape{6, 16});
    // A dynamic query must not hide a known f32-key versus f16-value mismatch.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    for (const auto i : {0, 1, 2, 11}) {
        inputs[i] = std::make_shared<Parameter>(element::i32, inputs[i].get_partial_shape());
    }
    // Matching integer data types still violate the floating-point attention contract.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_dynamic_ranks) {
    auto inputs = make_valid_inputs();
    for (auto& input : inputs) {
        input = std::make_shared<Parameter>(input.get_element_type(), PartialShape::dynamic());
    }
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // Even wholly unknown input ranks have a rank-two attention output by definition.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, -1}));
}

TEST(type_prop, paged_sparse_attention_individual_dynamic_ranks) {
    const auto inputs = make_valid_inputs();
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
        SCOPED_TRACE(input_index);
        auto dynamic_inputs = inputs;
        dynamic_inputs[input_index] =
            std::make_shared<Parameter>(inputs[input_index].get_element_type(), PartialShape::dynamic());
        const auto op = std::make_shared<PagedSparseAttention>(dynamic_inputs, 4);
        // Other inputs recover missing metadata, except the query width needed to derive query heads.
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, input_index == 0 ? -1 : 32}));
    }
}

TEST(type_prop, paged_sparse_attention_shape_inference_uses_supplied_shapes) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);
    auto shapes = ov::util::get_node_input_partial_shapes(*op);
    for (const auto i : {0, 1, 2, 9, 10}) {
        shapes[i][0] = 9;
    }
    // Inference must use the supplied nine-token shapes instead of the node's original six-token shapes.
    EXPECT_EQ(shape_infer(op.get(), shapes), (std::vector<PartialShape>{{9, 32}}));
    shapes[11] = PartialShape{1};
    // Supplied shapes must still enforce a scalar scale input.
    EXPECT_THROW(shape_infer(op.get(), shapes), NodeValidationFailure);
    shapes.pop_back();
    // Inference validates supplied arity before accessing any positional shape.
    EXPECT_THROW(shape_infer(op.get(), shapes), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_merges_intervals_and_symbols) {
    auto inputs = make_valid_inputs();
    PartialShape query_shape{Dimension(2, 10), 64};
    const auto symbols = set_shape_symbols(query_shape);
    inputs[0] = std::make_shared<Parameter>(element::f32, query_shape);
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension(4, 8), 32});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension(5, 7), 16});
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // Intersect all token intervals with the six-token selections instead of copying the query interval.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
    // Refining a token dimension must preserve its query symbol for downstream symbolic inference.
    EXPECT_EQ(op->get_output_partial_shape(0)[0].get_symbol(), symbols[0]);

    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension(2, 4), 32});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{Dimension(6, 8), 16});
    // Individually query-compatible intervals are invalid when they have no common token count.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_validates_each_rank_and_cache_type) {
    const auto inputs = make_valid_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(i);
        auto invalid = inputs;
        invalid[i] =
            std::make_shared<Parameter>(inputs[i].get_element_type(), i == 11 ? PartialShape{1} : PartialShape{});
        // Every port enforces its own rank; in particular, scale is scalar while all other ports are not.
        EXPECT_THROW(std::make_shared<PagedSparseAttention>(invalid, 4), NodeValidationFailure);
    }
    for (const auto i : {3, 4}) {
        for (const auto type : {element::i32, element::f64}) {
            auto invalid = inputs;
            invalid[i] = std::make_shared<Parameter>(type, inputs[i].get_partial_shape());
            // Neither integer i32 nor floating f64 is a supported cache storage format.
            EXPECT_THROW(std::make_shared<PagedSparseAttention>(invalid, 4), NodeValidationFailure);
        }
        auto dynamic = inputs;
        dynamic[i] = std::make_shared<Parameter>(element::dynamic, inputs[i].get_partial_shape());
        // An unresolved cache type is allowed until a later validation pass determines it.
        EXPECT_NO_THROW(std::make_shared<PagedSparseAttention>(dynamic, 4));
    }
    auto invalid = inputs;
    invalid[4] = std::make_shared<Parameter>(element::i8, inputs[4].get_partial_shape());
    // Signed i8 is key-cache-only and must not leak into the value-cache allowlist.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(invalid, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_rejects_inconsistent_dynamic_relationships) {
    auto inputs = make_valid_inputs();
    inputs[3] = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    inputs[4] = std::make_shared<Parameter>(element::bf16, PartialShape{8, 2, 30, 8});
    // A dynamic key cache must not hide the value cache's invalid page divisibility.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[5] = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    inputs[8] = std::make_shared<Parameter>(element::i32, PartialShape{5});
    // Offset arrays must agree even when past_lens cannot reveal the sequence count.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);

    inputs = make_valid_inputs();
    inputs[10] = std::make_shared<Parameter>(element::i32, PartialShape{6, 0});
    // A zero selection-head axis cannot describe a selection for any query head.
    EXPECT_THROW(std::make_shared<PagedSparseAttention>(inputs, 4), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_dynamic_head_product) {
    auto inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, -1});
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{6, -1});
    inputs[4] = std::make_shared<Parameter>(element::bf16, PartialShape{8, 2, 32, -1});
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // Unknown query-head count and value head size must not produce a spurious concrete output width.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, -1}));
    // Check dynamic status explicitly rather than only comparing the printed shape representation.
    EXPECT_TRUE(op->get_output_partial_shape(0)[1].is_dynamic());
    // Dynamic dimensions are intervals, so their product is dynamic rather than numeric (-1) * (-1).
    EXPECT_EQ(Dimension::dynamic() * Dimension::dynamic(), Dimension::dynamic());
}

TEST(type_prop, paged_sparse_attention_rejects_zero_head_dimensions) {
    const auto inputs = make_valid_inputs();
    for (const auto input_index : {0, 1, 2, 3, 4}) {
        SCOPED_TRACE(input_index);
        auto invalid_inputs = inputs;
        auto shape = inputs[input_index].get_partial_shape();
        shape[input_index < 3 ? 1 : 3] = 0;
        invalid_inputs[input_index] = std::make_shared<Parameter>(inputs[input_index].get_element_type(), shape);
        // Zero projection widths or cache head sizes cannot define positive attention head dimensions.
        EXPECT_THROW(std::make_shared<PagedSparseAttention>(invalid_inputs, 4), NodeValidationFailure);
    }
    for (const auto cache_index : {3, 4}) {
        auto invalid_inputs = inputs;
        auto shape = inputs[cache_index].get_partial_shape();
        shape[1] = 0;
        invalid_inputs[cache_index] = std::make_shared<Parameter>(inputs[cache_index].get_element_type(), shape);
        // Both caches require a positive KV-head count before any head-size division.
        EXPECT_THROW(std::make_shared<PagedSparseAttention>(invalid_inputs, 4), NodeValidationFailure);
    }
}

TEST(type_prop, paged_sparse_attention_accepts_zero_tokens) {
    auto inputs = make_valid_inputs();
    for (const auto input_index : {0, 1, 2, 9, 10}) {
        auto shape = inputs[input_index].get_partial_shape();
        shape[0] = 0;
        inputs[input_index] = std::make_shared<Parameter>(inputs[input_index].get_element_type(), shape);
    }
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // An empty token stream is valid and preserves the inferred per-token feature width.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{0, 32}));
}

TEST(type_prop, paged_sparse_attention_contract_dimensions) {
    const auto inputs = make_valid_inputs();
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    const auto shapes = ov::util::get_node_input_partial_shapes(*op);

    // Both caches share B, Hkv and Bs, while their last dimensions are independently S and Sv.
    for (const auto axis : {0, 1, 2}) {
        auto invalid = shapes;
        invalid[4][axis] = shapes[4][axis].get_length() + 1;
        EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);
    }
    // Key/value projections must equal Hkv*S and Hkv*Sv, respectively.
    for (const auto projection : {1, 2}) {
        auto invalid = shapes;
        invalid[projection][1] = shapes[projection][1].get_length() + 2;
        EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);
    }
    // Both sequence-offset arrays contain B_seq+1 entries, not B_seq or an unrelated length.
    for (const auto offsets : {6, 8}) {
        auto invalid = shapes;
        invalid[offsets][0] = 3;
        EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);
    }
    // Three selection heads do not divide the four query heads, even if both selection ports agree.
    auto invalid = shapes;
    invalid[9][1] = 3;
    invalid[10][1] = 3;
    EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);

    // Nonempty sinks have one entry per query head, not one per KV head.
    invalid = shapes;
    invalid[12][0] = 2;
    EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);

    // Hs may be any divisor of H; an empty selection still permits attention to the causal tail.
    for (const auto selection_heads : {1, 2, 4}) {
        auto valid = shapes;
        valid[9] = PartialShape{6, selection_heads, 0};
        valid[10] = PartialShape{6, selection_heads};
        valid[12] = PartialShape{0};
        EXPECT_EQ(shape_infer(op.get(), valid), (std::vector<PartialShape>{{6, 32}}));
    }
}

TEST(type_prop, paged_sparse_attention_derives_dynamic_cache_head_count) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);
    auto shapes = ov::util::get_node_input_partial_shapes(*op);
    shapes[3][1] = Dimension::dynamic();
    shapes[4][1] = Dimension::dynamic();
    // The projections and known head sizes still determine two KV heads when cache Hkv is unknown.
    EXPECT_EQ(shape_infer(op.get(), shapes), (std::vector<PartialShape>{{6, 32}}));

    for (const auto projection : {1, 2}) {
        auto invalid = shapes;
        invalid[projection][1] = shapes[projection][1].get_length() + 1;
        // An unknown Hkv must not allow a projection width indivisible by its known head size.
        EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);
    }
    auto invalid = shapes;
    invalid[2][1] = 24;
    // Key-derived Hkv=2 and value-derived Hkv=3 cannot describe the same KV caches.
    EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);

    invalid = shapes;
    invalid[0][1] = 80;
    // Inferred KV heads also constrain query grouping: five query heads cannot share two KV heads.
    EXPECT_THROW(shape_infer(op.get(), invalid), NodeValidationFailure);
}

TEST(type_prop, paged_sparse_attention_specification_example) {
    const OutputVector inputs{std::make_shared<Parameter>(element::bf16, PartialShape{300, 6144}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{300, 512}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{300, 512}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{-1, 2, 32, 256}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{-1, 2, 32, 256}),
                              std::make_shared<Parameter>(element::i32, PartialShape{-1}),
                              std::make_shared<Parameter>(element::i32, PartialShape{-1}),
                              std::make_shared<Parameter>(element::i32, PartialShape{-1}),
                              std::make_shared<Parameter>(element::i32, PartialShape{-1}),
                              std::make_shared<Parameter>(element::i32, PartialShape{300, 1, 512}),
                              std::make_shared<Parameter>(element::i32, PartialShape{300, 1}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{}),
                              std::make_shared<Parameter>(element::bf16, PartialShape{0})};
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // Reproduce the spec's XML example: 300 tokens, 24 query heads and Sv=256 produce [300, 6144].
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{300, 6144}));
    // The specification example uses BF16 attention independently of its dynamic physical block count.
    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
}

}  // namespace ov::testing
