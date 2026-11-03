// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <tuple>

#include "common_test_utils/test_assertions.hpp"
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
        EXPECT_NO_THROW(
            std::ignore =
                std::make_shared<PagedSparseAttention>(make_valid_inputs(element::f32, type, element::f16), 4));
    }
    for (const auto& type : value_cache_types) {
        // Value caches accept their specified storage types without forcing the key cache to match.
        EXPECT_NO_THROW(
            std::ignore =
                std::make_shared<PagedSparseAttention>(make_valid_inputs(element::f32, element::f16, type), 4));
    }
}

TEST(type_prop, paged_sparse_attention_rejects_invalid_key_block_size) {
    // Zero cannot define a selected block.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 0),
                    NodeValidationFailure,
                    ::testing::HasSubstr("key_block_size must be positive, but got 0."));
    // Negative block lengths cannot describe groups of key tokens.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(make_valid_inputs(), -4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("key_block_size must be positive, but got -4."));
}

TEST(type_prop, paged_sparse_attention_rejects_incompatible_shapes) {
    auto inputs = make_valid_inputs();
    inputs[1] = std::make_shared<Parameter>(element::f32, PartialShape{5, 32});
    // Current query and key projections must describe the same token stream.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("TDim::merge(tokens, tokens, key[0])"));

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 80});
    // Five query heads cannot be grouped evenly over two KV heads.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("!quotient.get_interval().empty()"));

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 64});
    inputs[1] = std::make_shared<Parameter>(element::f16, PartialShape{6, 32});
    // Projection types must match even when their shapes are otherwise valid.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("data_types_match"));

    inputs = make_valid_inputs();
    inputs[0] = std::make_shared<Parameter>(element::f32, PartialShape{6, 2, 32});
    // Query heads must be flattened into a rank-two projection.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("query.rank().compatible(2)"));

    inputs = make_valid_inputs();
    inputs[2] = std::make_shared<Parameter>(element::f32, PartialShape{6, 2, 8});
    // Value heads must also be flattened rather than exposed as a separate axis.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("value.rank().compatible(2)"));

    inputs = make_valid_inputs();
    inputs[10] = std::make_shared<Parameter>(element::i32, PartialShape{6, 2});
    // Selection counts must use the same selection-head axis as the selected indices.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("TDim::merge(selection_heads, selection_heads, selected_count[1])"));
}

TEST(type_prop, paged_sparse_attention_rejects_invalid_types_and_arity) {
    auto inputs = make_valid_inputs();
    inputs[11] = std::make_shared<Parameter>(element::f16, PartialShape{});
    // Scale shares the query/key/value type rather than having an independent precision.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("data_types_match"));

    inputs = make_valid_inputs();
    inputs[7] = std::make_shared<Parameter>(element::i64, PartialShape{12});
    // The page-table requires i32 indices, not merely any integer type.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("input_type.compatible(element::i32)"));

    inputs = make_valid_inputs();
    inputs.pop_back();
    // All 13 ports are required, including an explicit sinks input when it is empty.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("get_input_size() == 13"));
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
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("data_types_match"));

    for (const auto i : {0, 1, 2, 11}) {
        inputs[i] = std::make_shared<Parameter>(element::i32, inputs[i].get_partial_shape());
    }
    // Matching integer data types still violate the floating-point attention contract.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("data_type.is_dynamic() || data_type.is_real()"));
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
        // All three projection widths are required; cache shapes do not supply logical dimensions.
        EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, input_index < 3 ? -1 : 32}));
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
    OV_EXPECT_THROW(shape_infer(op.get(), shapes),
                    NodeValidationFailure,
                    ::testing::HasSubstr("scale.rank().compatible(0)"));
    shapes.pop_back();
    // Inference validates supplied arity before accessing any positional shape.
    OV_EXPECT_THROW(shape_infer(op.get(), shapes),
                    NodeValidationFailure,
                    ::testing::HasSubstr("input_shapes.size() == 13"));
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
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("TDim::merge(tokens, tokens, value[0])"));
}

TEST(type_prop, paged_sparse_attention_validates_each_rank_and_cache_type) {
    const auto inputs = make_valid_inputs();
    const char* rank_conditions[] = {"query.rank().compatible(2)",
                                     "key.rank().compatible(2)",
                                     "value.rank().compatible(2)",
                                     "key_cache.rank().is_dynamic()",
                                     "value_cache.rank().is_dynamic()",
                                     "past_lens.rank().compatible(1)",
                                     "subsequence_begins.rank().compatible(1)",
                                     "block_indices.rank().compatible(1)",
                                     "block_indices_begins.rank().compatible(1)",
                                     "selected_indices.rank().compatible(3)",
                                     "selected_count.rank().compatible(2)",
                                     "scale.rank().compatible(0)",
                                     "sinks.rank().compatible(1)"};
    for (size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(i);
        auto invalid = inputs;
        invalid[i] =
            std::make_shared<Parameter>(inputs[i].get_element_type(), i == 11 ? PartialShape{1} : PartialShape{});
        // Every port enforces its own rank; in particular, scale is scalar while all other ports are not.
        OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(invalid, 4),
                        NodeValidationFailure,
                        ::testing::HasSubstr(rank_conditions[i]));
    }
    for (const auto i : {3, 4}) {
        for (const auto type : {element::i32, element::f64}) {
            auto invalid = inputs;
            invalid[i] = std::make_shared<Parameter>(type, inputs[i].get_partial_shape());
            // Neither integer i32 nor floating f64 is a supported cache storage format.
            OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(invalid, 4),
                            NodeValidationFailure,
                            ::testing::HasSubstr(i == 3 ? "ov::util::contains(common_cache_types, key_cache_type)"
                                                        : "ov::util::contains(common_cache_types, value_cache_type)"));
        }
        auto dynamic = inputs;
        dynamic[i] = std::make_shared<Parameter>(element::dynamic, inputs[i].get_partial_shape());
        // An unresolved cache type is allowed until a later validation pass determines it.
        EXPECT_NO_THROW(std::ignore = std::make_shared<PagedSparseAttention>(dynamic, 4));
    }
    auto invalid = inputs;
    invalid[4] = std::make_shared<Parameter>(element::i8, inputs[4].get_partial_shape());
    // Signed i8 is key-cache-only and must not leak into the value-cache allowlist.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(invalid, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("ov::util::contains(common_cache_types, value_cache_type)"));
}

TEST(type_prop, paged_sparse_attention_rejects_inconsistent_dynamic_relationships) {
    auto inputs = make_valid_inputs();
    inputs[5] = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    inputs[8] = std::make_shared<Parameter>(element::i32, PartialShape{5});
    // Offset arrays must agree even when past_lens cannot reveal the sequence count.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("TDim::merge(offsets, offsets, block_indices_begins[0])"));

    inputs = make_valid_inputs();
    inputs[10] = std::make_shared<Parameter>(element::i32, PartialShape{6, 0});
    // A zero selection-head axis cannot describe a selection for any query head.
    OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                    NodeValidationFailure,
                    ::testing::HasSubstr("TDim::merge(selection_heads, selection_heads, selected_count[1])"));
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
    for (const auto input_index : {0, 1, 2}) {
        SCOPED_TRACE(input_index);
        auto invalid_inputs = inputs;
        auto shape = inputs[input_index].get_partial_shape();
        shape[1] = 0;
        invalid_inputs[input_index] = std::make_shared<Parameter>(inputs[input_index].get_element_type(), shape);
        // Logical projection widths remain positive even though cache dimensions are opaque.
        OV_EXPECT_THROW(std::ignore = std::make_shared<PagedSparseAttention>(invalid_inputs, 4),
                        NodeValidationFailure,
                        ::testing::HasSubstr(input_index == 0   ? "query[1].get_length() > 0"
                                             : input_index == 1 ? "key[1].get_length() > 0"
                                                                : "value[1].get_length() > 0"));
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

    // Both sequence-offset arrays contain B_seq+1 entries, not B_seq or an unrelated length.
    for (const auto offsets : {6, 8}) {
        auto invalid = shapes;
        invalid[offsets][0] = 3;
        OV_EXPECT_THROW(shape_infer(op.get(), invalid),
                        NodeValidationFailure,
                        ::testing::HasSubstr("TDim::merge(offsets, offsets, block_indices_begins[0])"));
    }
    // Selection-head divisibility needs logical H and is deferred to backend integration.
    auto invalid = shapes;
    invalid[9][1] = 3;
    invalid[10][1] = 3;
    EXPECT_EQ(shape_infer(op.get(), invalid), (std::vector<PartialShape>{{6, 32}}));

    // Sinks' logical head count likewise cannot be validated from opaque caches.
    invalid = shapes;
    invalid[12][0] = 2;
    EXPECT_EQ(shape_infer(op.get(), invalid), (std::vector<PartialShape>{{6, 32}}));

    // Hs may be any divisor of H; an empty selection still permits attention to the causal tail.
    for (const auto selection_heads : {1, 2, 4}) {
        auto valid = shapes;
        valid[9] = PartialShape{6, selection_heads, 0};
        valid[10] = PartialShape{6, selection_heads};
        valid[12] = PartialShape{0};
        EXPECT_EQ(shape_infer(op.get(), valid), (std::vector<PartialShape>{{6, 32}}));
    }
}

TEST(type_prop, paged_sparse_attention_projection_based_output) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);
    auto shapes = ov::util::get_node_input_partial_shapes(*op);
    shapes[3][1] = Dimension::dynamic();
    shapes[4][1] = Dimension::dynamic();
    // Output width does not need the cache's logical KV-head count.
    EXPECT_EQ(shape_infer(op.get(), shapes), (std::vector<PartialShape>{{6, 32}}));

    auto invalid = shapes;
    invalid[1][1] = 33;
    // Q/K must be integral for grouped-query attention, independently of storage layout.
    OV_EXPECT_THROW(shape_infer(op.get(), invalid),
                    NodeValidationFailure,
                    ::testing::HasSubstr("!quotient.get_interval().empty()"));
    auto valid = shapes;
    valid[2][1] = 17;
    // Value width is not constrained by an opaque cache's physical last axis.
    EXPECT_EQ(shape_infer(op.get(), valid), (std::vector<PartialShape>{{6, 34}}));
    valid = shapes;
    valid[2][1] = 24;
    // A different logical value head size changes output width without consulting caches.
    EXPECT_EQ(shape_infer(op.get(), valid), (std::vector<PartialShape>{{6, 48}}));
    invalid = shapes;
    invalid[0][1] = 80;
    // Q/K=2.5 cannot describe an integral query-to-KV head grouping.
    OV_EXPECT_THROW(shape_infer(op.get(), invalid),
                    NodeValidationFailure,
                    ::testing::HasSubstr("!quotient.get_interval().empty()"));
}

TEST(type_prop, paged_sparse_attention_output_width_overflow) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);
    auto shapes = ov::util::get_node_input_partial_shapes(*op);
    const auto max = std::numeric_limits<int64_t>::max() - 1;
    shapes[0][1] = 32;
    shapes[1][1] = 32;
    shapes[2][1] = max;
    // Divide first: Q*V overflows, but Q/K*V is exactly representable.
    EXPECT_EQ(shape_infer(op.get(), shapes), (std::vector<PartialShape>{{6, max}}));
    shapes[0][1] = 64;
    // Reject a genuinely overflowing output product instead of saturating/wrapping it.
    OV_EXPECT_THROW(shape_infer(op.get(), shapes),
                    NodeValidationFailure,
                    ::testing::HasSubstr("groups.get_length() <="));
    shapes[0][1] = 224;
    shapes[2][1] = 1317624576693539401LL;
    // INT64_MAX is the interval sentinel, not a finite output width.
    OV_EXPECT_THROW(shape_infer(op.get(), shapes),
                    NodeValidationFailure,
                    ::testing::HasSubstr("groups.get_length() <="));
}

TEST(type_prop, paged_sparse_attention_interval_width_division) {
    const auto op = std::make_shared<PagedSparseAttention>(make_valid_inputs(), 4);
    auto shapes = ov::util::get_node_input_partial_shapes(*op);
    shapes[1][1] = 2;
    shapes[2][1] = 1;
    shapes[0][1] = Dimension(63, 65);
    // Non-static widths retain interval division, selecting the compatible even width.
    EXPECT_EQ(shape_infer(op.get(), shapes), (std::vector<PartialShape>{{6, 32}}));
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

TEST(type_prop, paged_sparse_attention_compressed_cache_storage) {
    for (const auto key_type : {element::u4, element::u8, element::i8, element::f16}) {
        for (const auto value_type : {element::u4, element::u8, element::f16}) {
            for (const bool by_channel : {false, true}) {
                SCOPED_TRACE(key_type);
                SCOPED_TRACE(value_type);
                SCOPED_TRACE(by_channel);
                auto inputs = make_valid_inputs(element::f32, key_type, value_type);
                const int64_t key_params = key_type == element::u4   ? 16
                                           : key_type == element::i8 ? 4
                                           : key_type == element::u8 ? 8
                                                                     : 0;
                const int64_t value_params = value_type == element::u4 ? 16 : value_type == element::u8 ? 8 : 0;
                inputs[3] = std::make_shared<Parameter>(
                    key_type,
                    PartialShape{8, 2, by_channel ? 32 + key_params : 32, by_channel ? 16 : 16 + key_params});
                inputs[4] = std::make_shared<Parameter>(value_type, PartialShape{8, 2, 32, 8 + value_params});
                const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
                // Storage padding must not affect logical H=4, Sv=8 or output [T, H*Sv].
                EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
            }
        }
    }
}

TEST(type_prop, paged_sparse_attention_compressed_cache_revalidation) {
    const auto inputs = make_valid_inputs();
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 16);
    const auto key_cache = as_type_ptr<Parameter>(inputs[3].get_node_shared_ptr());
    const auto value_cache = as_type_ptr<Parameter>(inputs[4].get_node_shared_ptr());
    key_cache->set_element_type(element::u8);
    key_cache->set_partial_shape(PartialShape{8, 2, 40, 16});
    value_cache->set_element_type(element::u8);
    value_cache->set_partial_shape(PartialShape{8, 2, 32, 16});
    op->validate_and_infer_types();
    // Revalidation must not interpret padded key Bs=40 as logical Bs or padded value width=16 as Sv.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
    const auto cloned = op->clone_with_new_inputs(inputs);
    // Cloning preserves relaxed compressed-storage inference without rt_info or additional attributes.
    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(type_prop, paged_sparse_attention_opaque_cache_layouts) {
    for (const auto& type : {element::dynamic, element::f16, element::bf16, element::f32, element::u8, element::u4}) {
        for (const auto& key_shape :
             {PartialShape{0, 17}, PartialShape{2, 7, 3}, PartialShape{8, 16, 30, 0}, PartialShape{2, 3, 4, 5, 6}}) {
            for (const auto& value_shape :
                 {PartialShape{5, 9}, PartialShape{7, 11, 13}, PartialShape{9, 8, 7, 6}, PartialShape{1, 2, 3, 4, 5}}) {
                auto inputs = make_valid_inputs();
                inputs[3] = std::make_shared<Parameter>(type, key_shape);
                inputs[4] = std::make_shared<Parameter>(type, value_shape);
                const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
                // All supported cache ranks and arbitrary storage axes are independent of output inference.
                EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
            }
        }
    }
    for (const auto cache : {3, 4}) {
        for (const auto& shape : {PartialShape{}, PartialShape{1}, PartialShape{1, 2, 3, 4, 5, 6}}) {
            auto inputs = make_valid_inputs();
            inputs[cache] = std::make_shared<Parameter>(element::f16, shape);
            // PA-compatible cache containers still exclude ranks outside 2 through 5.
            OV_EXPECT_THROW(
                std::ignore = std::make_shared<PagedSparseAttention>(inputs, 4),
                NodeValidationFailure,
                ::testing::HasSubstr(cache == 3 ? "key_cache.rank().is_dynamic()" : "value_cache.rank().is_dynamic()"));
        }
    }
}

TEST(type_prop, paged_sparse_attention_compressed_cache_dynamic_heads) {
    auto inputs = make_valid_inputs(element::f32, element::u8, element::u4);
    inputs[3] = std::make_shared<Parameter>(element::u8, PartialShape{8, -1, 40, 24});
    inputs[4] = std::make_shared<Parameter>(element::u4, PartialShape{8, -1, 32, 24});
    const auto op = std::make_shared<PagedSparseAttention>(inputs, 4);
    // Projections determine output width even when compressed cache head counts are unknown.
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{6, 32}));
}

}  // namespace ov::testing
