// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/group_query_attention_decomposition.hpp"

namespace ov::test::intel_gpu {
namespace {

constexpr int64_t num_heads = 2;
constexpr int64_t kv_num_heads = 1;
constexpr int64_t head_size = 16;

std::shared_ptr<ov::Model> make_gqa_model(const ov::Dimension& past_len, bool causal = true) {
    const auto f32 = ov::element::f32;
    auto query = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, num_heads, 1, head_size});
    auto key = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, 1, head_size});
    auto past_key =
        std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, past_len, head_size});
    auto past_value =
        std::make_shared<ov::op::v0::Parameter>(f32, ov::PartialShape{1, kv_num_heads, past_len, head_size});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});

    ov::OutputVector inputs{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(inputs,
                                                                      num_heads,
                                                                      kv_num_heads,
                                                                      0.0f,
                                                                      false,
                                                                      false,
                                                                      0,
                                                                      ov::op::internal::GroupQueryAttentionQuantType::NONE,
                                                                      ov::op::internal::GroupQueryAttentionQuantType::NONE,
                                                                      -1,
                                                                      false,
                                                                      false,
                                                                      causal);
    ov::ResultVector results;
    for (const auto& output : gqa->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
    }
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{query,
                                                           key,
                                                           value,
                                                           past_key,
                                                           past_value,
                                                           seqlens_k,
                                                           total_sequence_length});
}

std::shared_ptr<ov::intel_gpu::op::SDPA> decompose_and_get_sdpa(const ov::Dimension& past_len, bool causal = true) {
    auto model = make_gqa_model(past_len, causal);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::GroupQueryAttentionDecomposition>();
    manager.run_passes(model);

    bool has_gqa = false;
    std::shared_ptr<ov::intel_gpu::op::SDPA> result;
    for (const auto& node : model->get_ordered_ops()) {
        has_gqa |= ov::is_type<ov::op::internal::GroupQueryAttention>(node);
        if (auto sdpa = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(node)) {
            result = sdpa;
        }
    }
    EXPECT_FALSE(has_gqa);
    return result;
}

TEST(GroupQueryAttentionDecompositionTest, uses_causal_sdpa_without_explicit_mask) {
    const auto sdpa = decompose_and_get_sdpa(ov::Dimension::dynamic());

    ASSERT_NE(sdpa, nullptr);
    ASSERT_EQ(sdpa->get_input_size(), 3u);
    EXPECT_TRUE(sdpa->get_causal());
    EXPECT_EQ(sdpa->get_causal_mask_alignment(), ov::intel_gpu::op::SDPA::CausalMaskAlignment::LOWER_RIGHT);

    const auto cloned = ov::as_type_ptr<ov::intel_gpu::op::SDPA>(sdpa->clone_with_new_inputs(sdpa->input_values()));
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->get_causal_mask_alignment(), ov::intel_gpu::op::SDPA::CausalMaskAlignment::LOWER_RIGHT);
}

TEST(GroupQueryAttentionDecompositionTest, builds_explicit_mask_for_bidirectional_attention) {
    const auto sdpa = decompose_and_get_sdpa(ov::Dimension::dynamic(), false);

    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 4u);
    EXPECT_FALSE(sdpa->get_causal());
}

}  // namespace
}  // namespace ov::test::intel_gpu
