// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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

namespace {
ov::OutputVector make_valid_pa_args(const element::Type& t = element::f32) {
    using ov::op::v0::Parameter;

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
    auto trig = std::make_shared<Parameter>(t, PartialShape{6});

    auto xthr = std::make_shared<Parameter>(t, PartialShape{});
    auto xbs = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto xst = std::make_shared<Parameter>(element::i32, PartialShape{});

    auto sinks = std::make_shared<Parameter>(t, PartialShape{1, 3, 4});

    auto arkv_start = std::make_shared<Parameter>(element::i32, PartialShape{});
    auto arkv_evict = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto arkv_idx = std::make_shared<Parameter>(element::i32, PartialShape{3});
    auto arkv_beg = std::make_shared<Parameter>(element::i32, PartialShape{3});

    return {query, key,   value, key_cache,  value_cache, past_lens, subseq,  block_idx, block_beg,
            scale, slide, alibi, maxctx,     scorew,      rotated,   deltas,  trig,      xthr,
            xbs,   xst,   sinks, arkv_start, arkv_evict,  arkv_idx,  arkv_beg};
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
                             adaptive_rkv_diversity_block_set_indices_begins};

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
                             adaptive_rkv_diversity_block_set_indices_begins};

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
                             adaptive_rkv_diversity_block_set_indices_begins};

    EXPECT_NO_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args));
}

TEST(type_prop, paged_attention_invalid_rank_query) {
    auto args = make_valid_pa_args();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3});  // query must be rank-2
    EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args), ov::NodeValidationFailure);
}

TEST(type_prop, paged_attention_invalid_type_scale) {
    auto args = make_valid_pa_args();
    args[9] = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});  // scale must be real type
    EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args), ov::NodeValidationFailure);
}

TEST(type_prop, paged_attention_invalid_rank_key_cache) {
    auto args = make_valid_pa_args();
    args[3] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{3, 4});  // key_cache must be rank 2..5
    EXPECT_THROW(std::ignore = std::make_shared<op::PagedAttentionExtension>(args), ov::NodeValidationFailure);
}

}  // namespace testing
}  // namespace ov
