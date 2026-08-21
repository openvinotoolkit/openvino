// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "npuw_transformations/decompose_gqa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/manager.hpp"

// Structural coverage for NPUW's DecomposeGQA (decompose_gqa.cpp): the internal GroupQueryAttention op must
// lower to a ScaledDotProductAttention-based subgraph in both the prefill and generate branches, with the
// sliding-window (local_window_size) band and the smooth_softmax / head_sink sink wired in. Numerical parity
// of the windowing coordinates is covered off-tree against a NumPy port of the ORT windowed attention; these
// tests guard that the pass keeps firing and emitting the expected structure after refactors / rebases.

namespace {

using ov::op::internal::GroupQueryAttention;
using ov::op::internal::GroupQueryAttentionQuantType;
using SDPA = ov::op::v13::ScaledDotProductAttention;

constexpr int64_t NUM_HEADS = 4;
constexpr int64_t KV_NUM_HEADS = 2;
constexpr int64_t HEAD_SIZE = 16;
constexpr int64_t CAPACITY = 8;

// Absent optional input: per #36842 the ONNX frontend now inserts an empty Constant (shape {0}) in place of
// a missing optional input; the decomposition detects it via ov::util::is_empty_constant_tensor.
auto make_absent_input(ov::element::Type t = ov::element::f32) {
    return ov::op::v0::Constant::create(t, ov::Shape{0}, {});
}

template <typename T>
size_t count_of(const std::shared_ptr<ov::Model>& model) {
    size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (ov::is_type<T>(op)) {
            ++n;
        }
    }
    return n;
}

std::shared_ptr<SDPA> first_sdpa(const std::shared_ptr<ov::Model>& model) {
    for (const auto& op : model->get_ops()) {
        if (auto sdpa = ov::as_type_ptr<SDPA>(op)) {
            return sdpa;
        }
    }
    return nullptr;
}

// Builds a single-GroupQueryAttention model. seq_len selects decode (1) vs multi-token; local_window_size < 0
// disables the window. head_sink adds the per-head sink input (index 11); smooth_softmax is an attribute.
std::shared_ptr<ov::Model> make_gqa_model(int64_t seq_len,
                                          int64_t local_window_size,
                                          bool smooth_softmax,
                                          bool head_sink) {
    const auto f32 = ov::element::f32;
    auto query = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{1, NUM_HEADS, size_t(seq_len), HEAD_SIZE});
    auto key = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{1, KV_NUM_HEADS, size_t(seq_len), HEAD_SIZE});
    auto value = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{1, KV_NUM_HEADS, size_t(seq_len), HEAD_SIZE});
    auto past_key = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{1, KV_NUM_HEADS, CAPACITY, HEAD_SIZE});
    auto past_value = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{1, KV_NUM_HEADS, CAPACITY, HEAD_SIZE});
    auto seqlens_k = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    auto total_sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});

    ov::ParameterVector params{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    ov::OutputVector args{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
    // cos_cache (7) / sin_cache (8): absent (do_rotary = false).
    args.push_back(make_absent_input());
    args.push_back(make_absent_input());
    if (head_sink) {
        // position_ids (9) / attention_bias (10): absent; head_sink is input 11.
        args.push_back(make_absent_input());
        args.push_back(make_absent_input());
        auto sink = std::make_shared<ov::op::v0::Parameter>(f32, ov::Shape{size_t(NUM_HEADS)});
        params.push_back(sink);
        args.push_back(sink);
    }

    auto gqa = std::make_shared<GroupQueryAttention>(args,
                                                     NUM_HEADS,
                                                     KV_NUM_HEADS,
                                                     /*scale*/ 0.0f,
                                                     /*do_rotary*/ false,
                                                     /*rotary_interleaved*/ false,
                                                     /*kv_cache_bit_width*/ 0,
                                                     /*k_quant_type*/ GroupQueryAttentionQuantType::NONE,
                                                     /*v_quant_type*/ GroupQueryAttentionQuantType::NONE,
                                                     local_window_size,
                                                     /*sliding_window_cache*/ false,
                                                     smooth_softmax);
    ov::ResultVector results;
    for (size_t i = 0; i < gqa->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
    }
    return std::make_shared<ov::Model>(results, params, "npuw_gqa");
}

void run_decompose(const std::shared_ptr<ov::Model>& model, bool is_prefill) {
    ov::pass::Manager manager;
    manager.register_pass<ov::npuw::DecomposeGQA>(is_prefill);
    manager.run_passes(model);
}

}  // namespace

// The generate branch (single-token decode) lowers to exactly one SDPA and removes the GQA op.
TEST(NpuwDecomposeGQA, GenerateDecomposesToSdpa) {
    auto model = make_gqa_model(/*seq_len*/ 1, /*window*/ -1, /*smooth*/ false, /*head_sink*/ false);
    run_decompose(model, /*is_prefill*/ false);
    EXPECT_EQ(count_of<GroupQueryAttention>(model), 0u);
    EXPECT_EQ(count_of<SDPA>(model), 1u);
}

// The prefill branch (multi-token) lowers the same way.
TEST(NpuwDecomposeGQA, PrefillDecomposesToSdpa) {
    auto model = make_gqa_model(/*seq_len*/ 4, /*window*/ -1, /*smooth*/ false, /*head_sink*/ false);
    run_decompose(model, /*is_prefill*/ true);
    EXPECT_EQ(count_of<GroupQueryAttention>(model), 0u);
    EXPECT_EQ(count_of<SDPA>(model), 1u);
}

// A sliding window (local_window_size >= 1) exercises the extra window-band masking in the generate branch;
// the op must still fully decompose.
TEST(NpuwDecomposeGQA, GenerateSlidingWindowDecomposes) {
    auto model = make_gqa_model(/*seq_len*/ 1, /*window*/ 2, /*smooth*/ false, /*head_sink*/ false);
    run_decompose(model, /*is_prefill*/ false);
    EXPECT_EQ(count_of<GroupQueryAttention>(model), 0u);
    EXPECT_EQ(count_of<SDPA>(model), 1u);
}

// Sliding window on the prefill branch (right-aligned frame path).
TEST(NpuwDecomposeGQA, PrefillSlidingWindowDecomposes) {
    auto model = make_gqa_model(/*seq_len*/ 4, /*window*/ 2, /*smooth*/ false, /*head_sink*/ false);
    run_decompose(model, /*is_prefill*/ true);
    EXPECT_EQ(count_of<GroupQueryAttention>(model), 0u);
    EXPECT_EQ(count_of<SDPA>(model), 1u);
}

// smooth_softmax wires an explicit sink into SDPA, selecting the 6-input form (Q, K, V, mask, scale, sink).
TEST(NpuwDecomposeGQA, GenerateSmoothSoftmaxWiresSdpaSink) {
    auto model = make_gqa_model(/*seq_len*/ 1, /*window*/ -1, /*smooth*/ true, /*head_sink*/ false);
    run_decompose(model, /*is_prefill*/ false);
    ASSERT_EQ(count_of<GroupQueryAttention>(model), 0u);
    auto sdpa = first_sdpa(model);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 6u);
}

// head_sink (input 11) is wired as the SDPA sink the same way.
TEST(NpuwDecomposeGQA, GenerateHeadSinkWiresSdpaSink) {
    auto model = make_gqa_model(/*seq_len*/ 1, /*window*/ -1, /*smooth*/ false, /*head_sink*/ true);
    run_decompose(model, /*is_prefill*/ false);
    ASSERT_EQ(count_of<GroupQueryAttention>(model), 0u);
    auto sdpa = first_sdpa(model);
    ASSERT_NE(sdpa, nullptr);
    EXPECT_EQ(sdpa->get_input_size(), 6u);
}
