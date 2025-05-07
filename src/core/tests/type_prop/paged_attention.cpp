// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace testing;

namespace ov::test {

class TypePropPagedAttentionInternalTest : public TypePropOpTest<op::PagedAttentionExtension> {};

TEST_F(TypePropPagedAttentionInternalTest, paged_attention_static_15_inputs) {
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
    const auto free_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{15});
    const auto max_blocks = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{7});

    const auto op = std::make_shared<op::PagedAttentionExtension>(query,
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
                                                                  free_block_indices,
                                                                  max_blocks);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}

TEST_F(TypePropPagedAttentionInternalTest, paged_attention_static_18_inputs_eviction_per_block) {
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

    const auto rotated_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto rotation_deltas = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{12, 1});
    const auto rotation_trig_lut = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{256, 4});

    const auto free_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{15});
    const auto max_blocks = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{7});

    const auto op = std::make_shared<op::PagedAttentionExtension>(query,
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
                                                                  rotated_block_indices,
                                                                  rotation_deltas,
                                                                  rotation_trig_lut,
                                                                  free_block_indices,
                                                                  max_blocks);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}

TEST_F(TypePropPagedAttentionInternalTest, paged_attention_static_18_inputs_eviction_per_token) {
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

    const auto rotated_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{3});
    const auto rotation_deltas = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{12, 5});
    const auto rotation_trig_lut = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{256, 4});

    const auto free_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{15});
    const auto max_blocks = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{7});

    const auto op = std::make_shared<op::PagedAttentionExtension>(query,
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
                                                                  rotated_block_indices,
                                                                  rotation_deltas,
                                                                  rotation_trig_lut,
                                                                  free_block_indices,
                                                                  max_blocks);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{3, 4}));
}

}  // namespace ov::test
