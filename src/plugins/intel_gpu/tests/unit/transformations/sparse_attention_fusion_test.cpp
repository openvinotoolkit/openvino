// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/sparse_pa.hpp"
#include "openvino/op/sparse_sdpa.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/sparse_attention_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr int64_t Di = 16;    // indexer head dim
constexpr int64_t Hidx = 2;   // indexer query heads
constexpr int64_t D = 32;     // hidden size
constexpr int64_t Hq = 4;     // main attention heads
constexpr int64_t Hkv = 2;    // paged kv heads (GQA)
constexpr int64_t dense_kv_heads = Hq;  // SDPA v13 needs K/V already expanded to Hq heads
constexpr int64_t Dh = 16;    // main K/V head size
constexpr int64_t r = 4;      // compress ratio
constexpr int64_t Rd = 8;     // indexer rotary dim (partial: Rd < Di)
constexpr int64_t max_positions = 64;
constexpr int64_t block_topk = 1;
constexpr float eps = 1e-5f;  // deliberately not the 1e-6 default
constexpr float q_gain_w = 0.5f;  // q leg uses an unfolded `1 + w` gain
// The pass runs after ConvertPrecision, so the data path is already f16.
const auto et = ov::element::f16;

std::shared_ptr<ov::op::v0::Constant> i64_const(std::vector<int64_t> v) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
}

std::shared_ptr<ov::op::v0::Constant> f32_scalar(float v) {
    return ov::op::v0::Constant::create(et, ov::Shape{}, {v});
}

ov::Output<ov::Node> last_axis_slice(const ov::Output<ov::Node>& x, int64_t begin, int64_t end, int64_t step = 1) {
    return std::make_shared<ov::op::v8::Slice>(x, i64_const({begin}), i64_const({end}), i64_const({step}), i64_const({-1}));
}

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::as_type_ptr<T>(node)) {
            count++;
        }
    }
    return count;
}

template <typename T>
std::shared_ptr<T> find_op(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (auto op = ov::as_type_ptr<T>(node)) {
            return op;
        }
    }
    return nullptr;
}

/// Skips layout-only ops (Convert/Reshape) between `out` and its real producer.
ov::Node* producer_through_layout(ov::Output<ov::Node> out) {
    auto* node = out.get_node();
    while (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node)) {
        node = node->get_input_node_ptr(0);
    }
    return node;
}

/// Decomposed RMSNorm: x / sqrt(mean(x^2) + eps) * gain.
ov::Output<ov::Node> rms_norm(const ov::Output<ov::Node>& x, const ov::Output<ov::Node>& gain) {
    auto sq = std::make_shared<ov::op::v1::Multiply>(x, x);
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, i64_const({-1}), true);
    auto add_eps = std::make_shared<ov::op::v1::Add>(mean, f32_scalar(eps));
    auto denom = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    auto normed = std::make_shared<ov::op::v1::Divide>(x, denom);
    return std::make_shared<ov::op::v1::Multiply>(normed, gain);
}

enum class RopeLayout { Halves, Interleaved };

/// `table[p, j] = 8 * p + j` (exact in f16) so a repack is observable by value.
std::vector<float> rope_table_values() {
    std::vector<float> values(max_positions * Rd);
    for (int64_t p = 0; p < max_positions; ++p) {
        for (int64_t j = 0; j < Rd; ++j) {
            values[p * Rd + j] = static_cast<float>(8 * p + j);
        }
    }
    return values;
}

/// Indexer sub-graph shared by the dense and paged builders: a shared `[(Hidx+1)*Di, D]`
/// projection over dynamic `[B, L, D]` tokens, decomposed RMSNorm on both legs, RoPE via
/// `Gather(table, position_ids)`, a ReadValue-backed raw-K history, static-`r` block-mean
/// pooling, and the ReLU -> ReduceSum -> 1/sqrt(Di) -> TopK score signature.
struct IndexerGraph {
    std::shared_ptr<ov::op::v0::Parameter> mixed;
    std::shared_ptr<ov::op::v0::Parameter> positions;
    std::shared_ptr<ov::op::v11::TopK> topk;
    std::shared_ptr<ov::op::v6::ReadValue> raw_k_read_value;
    std::shared_ptr<ov::op::v6::Assign> raw_k_assign;
};

IndexerGraph build_indexer_graph(RopeLayout layout) {
    IndexerGraph g;
    g.mixed = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, -1, D});
    g.positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});

    auto qk_weight = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>((Hidx + 1) * Di), static_cast<size_t>(D)},
                                                  std::vector<float>((Hidx + 1) * Di * D, 0.01f));
    auto qk_proj = std::make_shared<ov::op::v0::MatMul>(g.mixed, qk_weight, false, true);  // [B, L, (Hidx+1)*Di]

    auto q_heads = std::make_shared<ov::op::v1::Reshape>(last_axis_slice(qk_proj, 0, Hidx * Di), i64_const({0, 0, Hidx, Di}), true);
    auto q_w = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, q_gain_w));
    auto q_gain = std::make_shared<ov::op::v1::Add>(q_w, f32_scalar(1.0f));
    auto q_normed = rms_norm(q_heads, q_gain);

    auto k_heads = std::make_shared<ov::op::v1::Reshape>(last_axis_slice(qk_proj, Hidx * Di, (Hidx + 1) * Di), i64_const({0, 0, 1, Di}), true);
    auto k_gain = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, 1.0f));
    auto k_normed = rms_norm(k_heads, k_gain);

    auto table = ov::op::v0::Constant::create(et, ov::Shape{max_positions, Rd}, rope_table_values());
    auto rows = std::make_shared<ov::op::v8::Gather>(table, g.positions, i64_const({0}));  // [B, L, Rd]
    ov::Output<ov::Node> cos, sin;
    if (layout == RopeLayout::Halves) {
        cos = last_axis_slice(rows, 0, Rd / 2);
        sin = last_axis_slice(rows, Rd / 2, Rd);
    } else {
        cos = last_axis_slice(rows, 0, Rd, 2);
        sin = last_axis_slice(rows, 1, Rd, 2);
    }
    // Structural stand-in for the rotation: only the table/positions/layout proof matters here.
    auto cos_sin = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos, sin}, -1);
    auto rope_term = std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v1::ReduceSum>(cos_sin, i64_const({-1}), true), i64_const({2}));
    auto idx_q = std::make_shared<ov::op::v1::Add>(q_normed, rope_term);  // [B, L, Hidx, Di]
    auto new_k = std::make_shared<ov::op::v1::Add>(k_normed, rope_term);  // [B, L, 1, Di]

    auto raw_k_variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{-1, -1, Di}, et, "raw_k"});
    g.raw_k_read_value = std::make_shared<ov::op::v6::ReadValue>(raw_k_variable);
    auto new_k_flat = std::make_shared<ov::op::v1::Reshape>(new_k, i64_const({0, -1, Di}), true);
    auto raw_k_history = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g.raw_k_read_value, new_k_flat}, 1);  // [B, ?, Di]
    g.raw_k_assign = std::make_shared<ov::op::v6::Assign>(raw_k_history, raw_k_variable);

    auto pool_reshaped = std::make_shared<ov::op::v1::Reshape>(raw_k_history, i64_const({0, -1, r, Di}), true);  // [B, ?, r, Di]
    auto pooled = std::make_shared<ov::op::v1::ReduceMean>(pool_reshaped, i64_const({2}), false);               // [B, ?, Di]
    auto idx_k = std::make_shared<ov::op::v0::Unsqueeze>(pooled, i64_const({1}));                               // [B, 1, ?, Di]

    auto logits = std::make_shared<ov::op::v0::MatMul>(idx_q, idx_k, false, true);  // [B, L, Hidx, ?]
    auto relu = std::make_shared<ov::op::v0::Relu>(logits);
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(relu, i64_const({2}), false);  // [B, L, ?]
    auto scaled = std::make_shared<ov::op::v1::Multiply>(reduced, f32_scalar(1.0f / std::sqrt(static_cast<float>(Di))));
    g.topk = std::make_shared<ov::op::v11::TopK>(scaled, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {block_topk}), 2,
                                                 ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);
    return g;
}

/// Scalar zero that structurally depends on `topk_out`, added to `target` (or used as a
/// mask) to make the attention reachable from the indexer's selection.
ov::Output<ov::Node> zero_from_topk(const ov::Output<ov::Node>& topk_out) {
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(topk_out, ov::element::i64);
    auto shape_f = std::make_shared<ov::op::v0::Convert>(shape, et);
    auto scalar = std::make_shared<ov::op::v1::ReduceSum>(shape_f, i64_const({0}), false);
    return std::make_shared<ov::op::v1::Multiply>(scalar, f32_scalar(0.0f));
}

enum class DenseAttn { Causal, TopkMask, NonCausal, NonDefaultScale };

struct DenseGraph {
    std::shared_ptr<ov::Model> model;
    IndexerGraph indexer;
    std::shared_ptr<ov::op::v6::ReadValue> k_read;
};

DenseGraph build_dense_graph(DenseAttn attn, RopeLayout layout = RopeLayout::Halves) {
    DenseGraph out;
    out.indexer = build_indexer_graph(layout);
    auto& g = out.indexer;

    const ov::PartialShape kv_shape{-1, dense_kv_heads, -1, Dh};
    auto k_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, et, "main_k"});
    auto v_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, et, "main_v"});
    out.k_read = std::make_shared<ov::op::v6::ReadValue>(k_variable);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_variable);
    auto k_new = std::make_shared<ov::op::v0::Parameter>(et, kv_shape);
    auto v_new = std::make_shared<ov::op::v0::Parameter>(et, kv_shape);
    auto k_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{out.k_read, k_new}, 2);
    auto v_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{v_read, v_new}, 2);
    auto k_assign = std::make_shared<ov::op::v6::Assign>(k_full, k_variable);
    auto v_assign = std::make_shared<ov::op::v6::Assign>(v_full, v_variable);

    auto q_param = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hq, -1, Dh});
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    switch (attn) {
    case DenseAttn::Causal:
    case DenseAttn::NonCausal: {
        auto q_biased = std::make_shared<ov::op::v1::Add>(q_param, zero_from_topk(g.topk->output(0)));
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_biased, k_full, v_full, attn == DenseAttn::Causal);
        break;
    }
    case DenseAttn::TopkMask:
    case DenseAttn::NonDefaultScale: {
        const float scale = attn == DenseAttn::TopkMask ? 1.0f / std::sqrt(static_cast<float>(Dh)) : 0.5f;
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_param, k_full, v_full, zero_from_topk(g.topk->output(0)), f32_scalar(scale),
                                                                        false);
        break;
    }
    }
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);

    out.model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::SinkVector{k_assign, v_assign, g.raw_k_assign},
                                            ov::ParameterVector{g.mixed, g.positions, q_param, k_new, v_new});
    return out;
}

constexpr int64_t page_size = 16;

struct PagedAttentionBuild {
    std::shared_ptr<ov::op::PagedAttentionExtension> op;
    std::shared_ptr<ov::op::v0::Parameter> key_cache;
    std::shared_ptr<ov::op::v0::Parameter> value_cache;
};

/// Minimal 28-input `PagedAttentionExtension`; disabled optional inputs use the smallest
/// shapes its `validate_and_infer_types()` accepts.
PagedAttentionBuild build_paged_attention(const ov::Output<ov::Node>& query,
                                          const ov::Output<ov::Node>& key,
                                          const ov::Output<ov::Node>& value,
                                          int32_t sliding_window) {
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hkv, page_size, Dh});
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hkv, page_size, Dh});
    auto past_lens = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto subsequence_begins = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    auto block_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto block_indices_begins = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / std::sqrt(static_cast<float>(Dh))});
    auto i32_scalar0 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto i32_scalar1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto f32_empty = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    auto i32_empty = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    auto u8_empty = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{0}, {});

    ov::OutputVector args(28);
    args[0] = query;
    args[1] = key;
    args[2] = value;
    args[3] = key_cache;
    args[4] = value_cache;
    args[5] = past_lens;
    args[6] = subsequence_begins;
    args[7] = block_indices;
    args[8] = block_indices_begins;
    args[9] = scale;
    args[10] = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {sliding_window});
    args[11] = f32_empty;    // alibi_slopes
    args[12] = i32_scalar0;  // max_context_len
    args[13] = i32_scalar0;  // score_aggregation_window
    args[14] = i32_empty;    // rotated_block_indices
    args[15] = i32_empty;    // rotation_deltas
    args[16] = f32_empty;    // rotation_trig_lut
    args[17] = f32_empty;    // xattention_threshold
    args[18] = i32_scalar0;  // xattention_block_size
    args[19] = i32_scalar1;  // xattention_stride
    args[20] = f32_empty;    // sinks
    args[21] = i32_scalar0;  // adaptive_rkv_start_size
    args[22] = i32_empty;    // adaptive_rkv_evictable_sizes
    args[23] = i32_empty;    // adaptive_rkv_diversity_block_set_indices
    args[24] = i32_empty;    // adaptive_rkv_diversity_block_set_indices_begins
    args[25] = i32_empty;    // token_type_ids
    args[26] = u8_empty;     // qq_bias
    args[27] = i32_empty;    // qq_bias_begins
    return {std::make_shared<ov::op::PagedAttentionExtension>(args), key_cache, value_cache};
}

struct PagedGraph {
    std::shared_ptr<ov::Model> model;
    IndexerGraph indexer;
};

PagedGraph build_paged_graph(bool include_paged_attention, int32_t sliding_window = 0) {
    PagedGraph out;
    out.indexer = build_indexer_graph(RopeLayout::Halves);
    auto& g = out.indexer;

    auto q_param = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hq * Dh});
    auto k_param = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hkv * Dh});
    auto v_param = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, Hkv * Dh});

    ov::ParameterVector params{g.mixed, g.positions, q_param, k_param, v_param};
    ov::ResultVector results;
    if (include_paged_attention) {
        auto q_biased = std::make_shared<ov::op::v1::Add>(q_param, zero_from_topk(g.topk->output(0)));
        auto built = build_paged_attention(q_biased, k_param, v_param, sliding_window);
        params.push_back(built.key_cache);
        params.push_back(built.value_cache);
        results.push_back(std::make_shared<ov::op::v0::Result>(built.op->output(0)));
    } else {
        results.push_back(std::make_shared<ov::op::v0::Result>(g.topk->output(0)));
    }
    out.model = std::make_shared<ov::Model>(results, ov::SinkVector{g.raw_k_assign}, params);
    return out;
}

void expect_in_place_cache(const ov::Output<ov::Node>& input, ov::element::Type et) {
    auto param = ov::as_type_ptr<ov::op::v0::Parameter>(input.get_node_shared_ptr());
    ASSERT_NE(param, nullptr);
    EXPECT_EQ(param->get_element_type(), et);
    EXPECT_TRUE(param->output(0).get_rt_info().count(in_place_cache_rt_key));
    EXPECT_FALSE(param->output(0).get_names().empty());
}

void expect_norm_weight(const ov::Output<ov::Node>& input, float value) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
    ASSERT_NE(c, nullptr);
    for (auto v : c->cast_vector<float>()) {
        EXPECT_FLOAT_EQ(v, value);
    }
}

void expect_interleaved_table(const ov::Output<ov::Node>& input) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(input.get_node_shared_ptr());
    ASSERT_NE(c, nullptr);
    ASSERT_EQ(c->get_shape(), (ov::Shape{max_positions, Rd}));
    const auto values = c->cast_vector<float>();
    for (int64_t p = 0; p < max_positions; ++p) {
        for (int64_t j = 0; j < Rd / 2; ++j) {
            EXPECT_FLOAT_EQ(values[p * Rd + 2 * j], static_cast<float>(8 * p + j));
            EXPECT_FLOAT_EQ(values[p * Rd + 2 * j + 1], static_cast<float>(8 * p + Rd / 2 + j));
        }
    }
}

void run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<SparseAttentionFusion>();
    manager.run_passes(model);
}

}  // namespace

TEST(sparse_attention_fusion, dense_dynamic_causal_replaces_with_proven_inputs) {
    auto g = build_dense_graph(DenseAttn::Causal);
    const size_t params_before = g.model->get_parameters().size();
    run_fusion(g.model);

    ASSERT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 0u);
    auto indexer = find_op<ov::op::internal::QSAIndexer>(g.model);
    auto sparse_sdpa = find_op<ov::op::internal::SparseSDPA>(g.model);
    ASSERT_NE(indexer, nullptr);
    ASSERT_NE(sparse_sdpa, nullptr);
    EXPECT_EQ(g.model->get_parameters().size(), params_before + 3);
    // Only the raw-K Assign is retired; main K/V state keeps advancing.
    EXPECT_EQ(g.model->get_sinks().size(), 2u);

    using In = ov::op::internal::QSAIndexer::InputIdx;
    const auto& cfg = indexer->get_config();
    EXPECT_EQ(cfg.compress_ratio, static_cast<size_t>(r));
    EXPECT_EQ(cfg.block_topk, static_cast<size_t>(block_topk));
    EXPECT_EQ(cfg.indexer_head_dim, static_cast<size_t>(Di));
    EXPECT_EQ(cfg.indexer_rotary_dim, static_cast<size_t>(Rd));
    EXPECT_NEAR(cfg.eps, eps, 1e-7f);

    // Dynamic B/L survive: q_idx is [B, L, Hidx, Di], outputs keep dynamic L.
    EXPECT_EQ(indexer->get_input_partial_shape(In::Q_IDX), (ov::PartialShape{-1, -1, Hidx, Di}));
    EXPECT_EQ(indexer->get_input_partial_shape(In::K_IDX), (ov::PartialShape{-1, -1, 1, Di}));
    EXPECT_EQ(indexer->get_output_partial_shape(0), (ov::PartialShape{-1, 1, -1, block_topk}));

    expect_norm_weight(indexer->input_value(In::Q_NORM_WEIGHT), 1.0f + q_gain_w);
    expect_norm_weight(indexer->input_value(In::K_NORM_WEIGHT), 1.0f);
    expect_interleaved_table(indexer->input_value(In::ROTARY_COS_SIN));

    // position_ids are the model's own RoPE gather indices, [B, L] i32.
    EXPECT_EQ(indexer->get_input_element_type(In::POSITION_IDS), ov::element::i32);
    EXPECT_EQ(indexer->get_input_partial_shape(In::POSITION_IDS).rank(), 2);
    EXPECT_EQ(producer_through_layout(indexer->input_value(In::POSITION_IDS)), g.indexer.positions.get());

    // past_lens [B] i32 comes from the main K history, not the retired raw-K Variable.
    EXPECT_EQ(indexer->get_input_element_type(In::PAST_LENS), ov::element::i32);
    auto broadcast = producer_through_layout(indexer->input_value(In::PAST_LENS));
    auto gather = broadcast->get_input_node_ptr(0);
    auto shape_of = ov::as_type<ov::op::v3::ShapeOf>(gather->get_input_node_ptr(0));
    ASSERT_NE(shape_of, nullptr);
    EXPECT_EQ(shape_of->get_input_node_ptr(0), g.k_read.get());

    expect_in_place_cache(indexer->input_value(In::INDEXER_RAW_K_CACHE), ov::element::f16);
    expect_in_place_cache(indexer->input_value(In::SUMMARY_CACHE), ov::element::f16);
    expect_in_place_cache(indexer->input_value(In::INDEXER_POSITION_CACHE), ov::element::i32);
    EXPECT_EQ(indexer->get_input_partial_shape(In::INDEXER_RAW_K_CACHE), (ov::PartialShape{-1, 1, -1, Di}));

    const auto& attn_cfg = sparse_sdpa->get_config();
    EXPECT_EQ(attn_cfg.num_heads, static_cast<size_t>(Hq));
    EXPECT_EQ(attn_cfg.num_kv_heads, static_cast<size_t>(dense_kv_heads));
    EXPECT_EQ(attn_cfg.k_head_size, static_cast<size_t>(Dh));
    EXPECT_EQ(attn_cfg.sel_block_size, static_cast<size_t>(r));
    EXPECT_EQ(sparse_sdpa->get_input_size(), ov::op::internal::SparseSDPA::InputIdx::COUNT_MIN);
}

TEST(sparse_attention_fusion, dense_accepts_topk_selection_mask_and_interleaved_table) {
    auto g = build_dense_graph(DenseAttn::TopkMask, RopeLayout::Interleaved);
    run_fusion(g.model);

    ASSERT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 0u);
    auto indexer = find_op<ov::op::internal::QSAIndexer>(g.model);
    ASSERT_NE(indexer, nullptr);
    // An already-interleaved table is passed through unchanged (same values as the source).
    auto table = ov::as_type_ptr<ov::op::v0::Constant>(indexer->get_input_node_shared_ptr(ov::op::internal::QSAIndexer::InputIdx::ROTARY_COS_SIN));
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(table->cast_vector<float>(), rope_table_values());
}

TEST(sparse_attention_fusion, dense_bails_out_on_unrepresentable_attention) {
    for (auto attn : {DenseAttn::NonCausal, DenseAttn::NonDefaultScale}) {
        auto g = build_dense_graph(attn);
        const size_t params_before = g.model->get_parameters().size();
        run_fusion(g.model);

        EXPECT_EQ(count_ops<ov::op::internal::QSAIndexer>(g.model), 0u);
        EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 1u);
        EXPECT_EQ(g.model->get_parameters().size(), params_before);
        EXPECT_EQ(g.model->get_sinks().size(), 3u);
    }
}

TEST(sparse_attention_fusion, paged_bails_out_without_reachable_paged_attention) {
    auto g = build_paged_graph(/*include_paged_attention=*/false);
    const size_t params_before = g.model->get_parameters().size();
    run_fusion(g.model);

    EXPECT_EQ(count_ops<ov::op::internal::PagedQSAIndexer>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::internal::SparsePA>(g.model), 0u);
    EXPECT_EQ(g.model->get_parameters().size(), params_before);
}

TEST(sparse_attention_fusion, paged_bails_out_on_sliding_window) {
    auto g = build_paged_graph(/*include_paged_attention=*/true, /*sliding_window=*/8);
    run_fusion(g.model);

    EXPECT_EQ(count_ops<ov::op::internal::SparsePA>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::PagedAttentionExtension>(g.model), 1u);
}

TEST(sparse_attention_fusion, paged_replaces_with_proven_inputs) {
    auto g = build_paged_graph(/*include_paged_attention=*/true);
    const size_t params_before = g.model->get_parameters().size();
    run_fusion(g.model);

    ASSERT_EQ(count_ops<ov::op::PagedAttentionExtension>(g.model), 0u);
    auto indexer = find_op<ov::op::internal::PagedQSAIndexer>(g.model);
    auto sparse_pa = find_op<ov::op::internal::SparsePA>(g.model);
    ASSERT_NE(indexer, nullptr);
    ASSERT_NE(sparse_pa, nullptr);
    EXPECT_EQ(g.model->get_parameters().size(), params_before + 3);

    using In = ov::op::internal::PagedQSAIndexer::InputIdx;
    const auto& cfg = indexer->get_config();
    EXPECT_EQ(cfg.compress_ratio, static_cast<size_t>(r));
    EXPECT_EQ(cfg.page_size, static_cast<size_t>(page_size));
    EXPECT_EQ(cfg.indexer_rotary_dim, static_cast<size_t>(Rd));
    EXPECT_NEAR(cfg.eps, eps, 1e-7f);

    EXPECT_EQ(indexer->get_input_partial_shape(In::Q_IDX), (ov::PartialShape{-1, Hidx * Di}));
    EXPECT_EQ(indexer->get_input_partial_shape(In::K_IDX), (ov::PartialShape{-1, Di}));
    EXPECT_EQ(indexer->get_input_element_type(In::POSITION_IDS), ov::element::i32);
    EXPECT_EQ(indexer->get_input_partial_shape(In::POSITION_IDS).rank(), 1);
    EXPECT_EQ(producer_through_layout(indexer->input_value(In::POSITION_IDS)), g.indexer.positions.get());
    expect_interleaved_table(indexer->input_value(In::ROTARY_COS_SIN));
    expect_norm_weight(indexer->input_value(In::Q_NORM_WEIGHT), 1.0f + q_gain_w);

    expect_in_place_cache(indexer->input_value(In::INDEXER_RAW_K_CACHE), ov::element::f16);
    expect_in_place_cache(indexer->input_value(In::SUMMARY_CACHE), ov::element::f16);
    expect_in_place_cache(indexer->input_value(In::INDEXER_POSITION_CACHE), ov::element::i32);
    EXPECT_EQ(indexer->get_input_partial_shape(In::INDEXER_RAW_K_CACHE), (ov::PartialShape{-1, 1, page_size, Di}));
    EXPECT_EQ(indexer->get_input_partial_shape(In::INDEXER_POSITION_CACHE), (ov::PartialShape{-1, page_size / r}));

    const auto& attn_cfg = sparse_pa->get_config();
    EXPECT_EQ(attn_cfg.num_heads, static_cast<size_t>(Hq));
    EXPECT_EQ(attn_cfg.num_kv_heads, static_cast<size_t>(Hkv));
    EXPECT_EQ(attn_cfg.block_size, static_cast<size_t>(page_size));
    EXPECT_EQ(attn_cfg.sel_block_size, static_cast<size_t>(r));
    EXPECT_FLOAT_EQ(attn_cfg.scale, 1.0f / std::sqrt(static_cast<float>(Dh)));
}
