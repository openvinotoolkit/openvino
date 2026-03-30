// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <tuple>
#include <utility>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace testing {
using ::testing::HasSubstr;

namespace {
ov::OutputVector make_valid_pa_args(const element::Type& t = element::f32) {
    using ov::op::v0::Parameter;

    // rotation_trig_lut and xattention_threshold only accept f16/f32 per the spec.
    // For bf16 tests, pin these two inputs to f32 instead.
    const auto trig_t = (t == element::bf16) ? element::f32 : t;

    auto query = std::make_shared<Parameter>(t, PartialShape{3, 4});
    auto key = std::make_shared<Parameter>(t, PartialShape{3, 4});
    auto value = std::make_shared<Parameter>(t, PartialShape{3, 4});

    auto key_cache = std::make_shared<Parameter>(t, PartialShape{4, 3, 32, 4});
    auto value_cache = std::make_shared<Parameter>(t, PartialShape{4, 3, 32, 4});

    auto past_lens = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto subseq = std::make_shared<Parameter>(element::i32, PartialShape{4});
    auto block_idx = std::make_shared<Parameter>(element::i32, PartialShape{6});
    auto block_beg = std::make_shared<Parameter>(element::i32, PartialShape{4});

    auto scale = std::make_shared<Parameter>(t, PartialShape{});
    auto slide = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto alibi = std::make_shared<Parameter>(t, PartialShape{3});
    auto maxctx = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto scorew = std::make_shared<Parameter>(element::i32, PartialShape{});

    auto rotated = std::make_shared<Parameter>(element::i32, PartialShape{6});
    auto deltas = std::make_shared<Parameter>(element::i32, PartialShape{6});
    auto trig = std::make_shared<Parameter>(trig_t, PartialShape{6});

    auto xthr = std::make_shared<Parameter>(trig_t, PartialShape{1});
    auto xbs = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto xst = std::make_shared<Parameter>(element::i32, PartialShape{});

    auto sinks = std::make_shared<Parameter>(t, PartialShape{1, 3, 1, 1});

    auto arkv_start = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto arkv_evict = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto arkv_idx = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto arkv_beg = std::make_shared<Parameter>(element::i32, PartialShape{3});

    auto token_type_ids = std::make_shared<Parameter>(element::i32, PartialShape{0});

    return {query, key,   value, key_cache,  value_cache, past_lens, subseq,   block_idx,     block_beg,
            scale, slide, alibi, maxctx,     scorew,      rotated,   deltas,   trig,          xthr,
            xbs,   xst,   sinks, arkv_start, arkv_evict,  arkv_idx,  arkv_beg, token_type_ids};
}
}  // namespace
TEST(type_prop, paged_attention_static_eviction_per_block) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, 2, 5, 4});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, 2, 5, 4});
    const auto past_lens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{15});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{8});
    const auto scale = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
    const auto sliding_window = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto alibi_slopes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{9});
    const auto max_context_len = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto score_aggregation_window = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});

    const auto rotated_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto rotation_deltas = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{12, 1});
    const auto rotation_trig_lut = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{256, 4});

    const auto xattention_threshold = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{5});
    const auto xattention_block_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto xattention_stride = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    const auto sinks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 1, 1});

    const auto adaptive_rkv_start_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto adaptive_rkv_evictable_sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto adaptive_rkv_diversity_block_set_indices =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{10});
    const auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});

    const auto token_type_ids = std::make_shared<op::v0::Parameter>(ov::element::i32, ov::Shape{0});

    ov::OutputVector args = {query,
                             key,
                             value,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scale,
                             sliding_window,
                             alibi_slopes,
                             max_context_len,
                             score_aggregation_window,
                             rotated_block_indices,
                             rotation_deltas,
                             rotation_trig_lut,
                             xattention_threshold,
                             xattention_block_size,
                             xattention_stride,
                             sinks,
                             adaptive_rkv_start_size,
                             adaptive_rkv_evictable_sizes,
                             adaptive_rkv_diversity_block_set_indices,
                             adaptive_rkv_diversity_block_set_indices_begins,
                             token_type_ids};

    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}

TEST(type_prop, paged_attention_static_eviction_per_token) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto key_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, 2, 5, 4});
    const auto value_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, 2, 5, 4});
    const auto past_lens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{15});
    const auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{8});
    const auto scale = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{});
    const auto sliding_window = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto alibi_slopes = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{9});
    const auto max_context_len = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto score_aggregation_window = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});

    const auto rotated_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto rotation_deltas = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{12, 5});
    const auto rotation_trig_lut = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{256, 4});

    const auto xattention_threshold = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{5});
    const auto xattention_block_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto xattention_stride = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    const auto sinks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 1, 1});

    const auto adaptive_rkv_start_size = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});
    const auto adaptive_rkv_evictable_sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});
    const auto adaptive_rkv_diversity_block_set_indices =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{10});
    const auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<op::v0::Parameter>(element::i32, PartialShape{5});

    const auto token_type_ids = std::make_shared<op::v0::Parameter>(ov::element::i32, ov::Shape{0});

    ov::OutputVector args = {query,
                             key,
                             value,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scale,
                             sliding_window,
                             alibi_slopes,
                             max_context_len,
                             score_aggregation_window,
                             rotated_block_indices,
                             rotation_deltas,
                             rotation_trig_lut,
                             xattention_threshold,
                             xattention_block_size,
                             xattention_stride,
                             sinks,
                             adaptive_rkv_start_size,
                             adaptive_rkv_evictable_sizes,
                             adaptive_rkv_diversity_block_set_indices,
                             adaptive_rkv_diversity_block_set_indices_begins,
                             token_type_ids};

    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}

TEST(type_prop, paged_attention_dynamic_ranks_and_types) {
    using namespace ov::op;
    const auto dyn = PartialShape::dynamic();

    const auto query = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto key = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto value = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto key_cache = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto value_cache = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto past_lens = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto subsequence_begins = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto block_indices = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto block_indices_begins = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto scale = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto sliding_window = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto alibi_slopes = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto max_context_len = std::make_shared<v0::Parameter>(element::dynamic, dyn);
    const auto score_aggregation_window = std::make_shared<v0::Parameter>(element::dynamic, dyn);

    const auto rotated_block_indices = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto rotation_deltas = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto rotation_trig_lut = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);

    const auto xattention_threshold = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto xattention_block_size = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto xattention_stride = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto sinks = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 1, 1});

    const auto adaptive_rkv_start_size = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto adaptive_rkv_evictable_sizes = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto adaptive_rkv_diversity_block_set_indices = std::make_shared<op::v0::Parameter>(element::dynamic, dyn);
    const auto adaptive_rkv_diversity_block_set_indices_begins =
        std::make_shared<op::v0::Parameter>(element::dynamic, dyn);

    const auto token_type_ids = std::make_shared<op::v0::Parameter>(ov::element::i32, ov::Shape{0});

    ov::OutputVector args = {query,
                             key,
                             value,
                             key_cache,
                             value_cache,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             scale,
                             sliding_window,
                             alibi_slopes,
                             max_context_len,
                             score_aggregation_window,
                             rotated_block_indices,
                             rotation_deltas,
                             rotation_trig_lut,
                             xattention_threshold,
                             xattention_block_size,
                             xattention_stride,
                             sinks,
                             adaptive_rkv_start_size,
                             adaptive_rkv_evictable_sizes,
                             adaptive_rkv_diversity_block_set_indices,
                             adaptive_rkv_diversity_block_set_indices_begins,
                             token_type_ids};

    EXPECT_NO_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args));
}

TEST(type_prop, paged_attention_invalid_rank_query) {
    auto args = make_valid_pa_args();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});  // query must be rank-2
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("Rank of `query`"));
}

TEST(type_prop, paged_attention_invalid_type_scale) {
    auto args = make_valid_pa_args();
    args[9] = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});  // scale must be real type
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("Element type of `scale`"));
}

TEST(type_prop, paged_attention_invalid_rank_key_cache) {
    auto args = make_valid_pa_args();
    args[3] = std::make_shared<op::v0::Parameter>(element::f32,
                                                  PartialShape{3});  // key_cache allows rank 2..5; rank 1 is invalid
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("Rank of `key_cache`"));
}

TEST(type_prop, paged_attention_invalid_input_count) {
    auto args = make_valid_pa_args();
    args.pop_back();  // 24 instead of 25
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("26 inputs"));
}

TEST(type_prop, paged_attention_invalid_past_lens_rank) {
    auto args = make_valid_pa_args();
    args[5] = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3, 1});  // past_lens must be rank-1
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("Rank of `past_lens`"));
}

TEST(type_prop, paged_attention_invalid_type_past_lens) {
    auto args = make_valid_pa_args();
    args[5] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});  // past_lens must be i32
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("Element type of `past_lens`"));
}

// ---------- dtype parametrisation ----------
class PagedAttentionTypePropDtype : public ::testing::TestWithParam<element::Type> {};

TEST_P(PagedAttentionTypePropDtype, OutputTypeMatchesInput) {
    const auto t = GetParam();
    auto args = make_valid_pa_args(t);
    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_element_type(0), t);
    EXPECT_EQ(op->get_output_element_type(1), t);
    EXPECT_EQ(op->get_output_element_type(2), t);
}

INSTANTIATE_TEST_SUITE_P(type_prop,
                         PagedAttentionTypePropDtype,
                         ::testing::Values(element::f32, element::bf16, element::f16),
                         [](const ::testing::TestParamInfo<element::Type>& info) {
                             return info.param.get_type_name();
                         });

// ---------- GQA scenarios ----------
TEST(type_prop, paged_attention_gqa_output_shape) {
    // GQA: q_heads=6, kv_heads=2, head_size=4 => q=[B,24], k=[B,8], v=[B,8]
    // Output features = q * v / k = 24 * 8 / 8 = 24  (= q_heads * v_head_size)
    auto args = make_valid_pa_args();
    const auto B = 3;
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{B, 24});        // Q: 6 heads * 4
    args[1] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{B, 8});         // K: 2 heads * 4
    args[2] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{B, 8});         // V: 2 heads * 4
    args[3] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4, 2, 32, 4});  // key_cache
    args[4] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{4, 2, 32, 4});  // value_cache
    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{B, 24}));
}

TEST(type_prop, paged_attention_gqa_mismatched_divisibility) {
    // q * v / k must be integer; here q=5, k=4, v=4 => 5*4/4=5 (ok but unusual)
    auto args = make_valid_pa_args();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 5});
    args[1] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    args[2] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 5}));
}

TEST(type_prop, paged_attention_output_feature_not_divisible) {
    // q=5, k=3, v=4 => 5*4 = 20, 20 % 3 != 0 => should throw
    auto args = make_valid_pa_args();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 5});
    args[1] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 3});
    args[2] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args),
                    ov::NodeValidationFailure,
                    HasSubstr("divisible"));
}

// ---------- output 1 & 2 validation ----------
TEST(type_prop, paged_attention_outputs_1_and_2_shapes) {
    auto args = make_valid_pa_args();
    const auto op = std::make_shared<op::PagedAttentionExtension>(args);
    // Output 0: [B_token, out_features]
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
    // Output 1 (scores): 1-D, dynamic without constant past_lens
    EXPECT_EQ(op->get_output_partial_shape(1).rank().get_length(), 1);
    // Output 2 (diversity): 1-D
    EXPECT_EQ(op->get_output_partial_shape(2).rank().get_length(), 1);
    // All outputs share query element type
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::f32);
}

}  // namespace testing
}  // namespace ov
