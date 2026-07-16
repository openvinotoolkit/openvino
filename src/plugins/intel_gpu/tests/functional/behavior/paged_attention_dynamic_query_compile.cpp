// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {

// Build a minimal PA model that reproduces the gemma-4-e2b crash scenario:
// query input is fully dynamic [?, ?] and the reshape input producer is 2D.
// Without num_q_heads in rt_info, CreatePagedAttentionExtensionOp crashes
// trying to access reshape_input[2] on a rank-2 shape.
std::shared_ptr<ov::Model> make_pa_model_dynamic_query(bool set_num_q_heads_rt_info) {
    const int64_t num_q_heads = 8;
    const int64_t k_head_size = 256;
    const int64_t num_kv_heads = 4;
    const int64_t v_head_size = 256;

    // Reshape with 2D input to simulate CommonOptimizations output.
    // PA query becomes fully dynamic [?, ?] — same as gemma-4-e2b PA[1]+.
    auto reshape_input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1});
    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, -1});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(reshape_input, reshape_shape, true);

    auto pa_key = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{-1, num_kv_heads * k_head_size});
    auto pa_value = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{-1, num_kv_heads * v_head_size});
    auto pa_key_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{-1, num_kv_heads, k_head_size, 16});
    auto pa_value_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f16, ov::PartialShape{-1, num_kv_heads, 16, v_head_size});

    auto pa_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_subseq_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pa_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / std::sqrt(float(k_head_size))});
    auto pa_sliding_window = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto pa_alibi_slopes = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    auto pa_max_context_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    auto pa_score_agg_window = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto pa_rotated_block_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_rotation_deltas = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_rotation_trig_lut = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{0}, {});
    auto pa_xattn_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    auto pa_xattn_block_size = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_xattn_stride = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_sinks = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{0}, {});
    auto pa_arkv_start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto pa_arkv_evict = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_arkv_div_idx = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_arkv_div_begins = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_token_type_ids = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto pa_qq_bias = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{0}, {});
    auto pa_qq_bias_begins = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});

    ov::OutputVector pa_args = {
        reshape->output(0),                                                 // 0: query (from 2D reshape)
        pa_key, pa_value, pa_key_cache, pa_value_cache,                     // 1-4
        pa_past_lens, pa_subseq_begins, pa_block_indices,                   // 5-7
        pa_block_indices_begins, pa_scale, pa_sliding_window,               // 8-10
        pa_alibi_slopes, pa_max_context_len, pa_score_agg_window,           // 11-13
        pa_rotated_block_indices, pa_rotation_deltas, pa_rotation_trig_lut, // 14-16
        pa_xattn_threshold, pa_xattn_block_size, pa_xattn_stride,           // 17-19
        pa_sinks, pa_arkv_start, pa_arkv_evict,                             // 20-22
        pa_arkv_div_idx, pa_arkv_div_begins, pa_token_type_ids,             // 23-25
        pa_qq_bias, pa_qq_bias_begins                                        // 26-27
    };

    auto pa_node = std::make_shared<ov::op::PagedAttentionExtension>(pa_args);

    pa_node->get_rt_info()["num_k_heads"] = num_kv_heads;
    pa_node->get_rt_info()["k_head_size"] = k_head_size;
    pa_node->get_rt_info()["num_v_heads"] = num_kv_heads;
    pa_node->get_rt_info()["v_head_size"] = v_head_size;

    if (set_num_q_heads_rt_info) {
        pa_node->get_rt_info()["num_q_heads"] = num_q_heads;
    }

    auto result = std::make_shared<ov::op::v0::Result>(pa_node->output(0));

    return std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{reshape_input, pa_key, pa_value, pa_key_cache, pa_value_cache,
                            pa_past_lens, pa_subseq_begins, pa_block_indices,
                            pa_block_indices_begins, pa_max_context_len});
}

class PagedAttentionDynamicQueryTest : public ::testing::Test {};

// Reproduces gemma-4-e2b crash: PA query is fully dynamic [?, ?] with 2D reshape input.
// With num_q_heads in rt_info, compile must succeed (CVS-184666 fix).
TEST_F(PagedAttentionDynamicQueryTest, smoke_CompileDynamicQueryWithRtInfo) {
    auto model = make_pa_model_dynamic_query(true);

    ov::Core core;
    ov::CompiledModel compiled_model;
    ASSERT_NO_THROW(compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU));
}

// Without num_q_heads in rt_info and a 2D reshape input, compile must throw
// a clear assertion rather than crashing with "Accessing out-of-range dimension".
TEST_F(PagedAttentionDynamicQueryTest, smoke_CompileDynamicQueryWithoutRtInfoThrows) {
    auto model = make_pa_model_dynamic_query(false);

    ov::Core core;
    ASSERT_ANY_THROW(core.compile_model(model, ov::test::utils::DEVICE_GPU));
}

}  // namespace
