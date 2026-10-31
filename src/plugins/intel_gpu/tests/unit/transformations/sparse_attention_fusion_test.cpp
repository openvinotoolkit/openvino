// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/sparse_attention_fusion.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/paged_qsa_indexer.hpp"
#include "intel_gpu/op/qsa_indexer.hpp"
#include "intel_gpu/op/sparse_pa.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

constexpr int64_t Di = 16;           // indexer head dim
constexpr int64_t Hi = 2;            // indexer query heads
constexpr int64_t Dr = 8;            // indexer rotary dim (partial: Dr < Di)
constexpr int64_t r = 4;             // key block size
constexpr int64_t token_budget = 8;  // two blocks per selection
constexpr int64_t block_topk = token_budget / r;
constexpr int64_t Hq = 4;     // attention heads
constexpr int64_t Hkv = 2;    // key/value heads
constexpr int64_t Dh = 16;    // attention head size
constexpr float eps = 1e-5f;  // deliberately not the 1e-6 default

constexpr int64_t pages = 8;
constexpr int64_t page_size = 16;
constexpr int64_t pages_blocks = page_size / r;

std::shared_ptr<ov::op::v0::Parameter> param(const ov::PartialShape& shape, ov::element::Type et = ov::element::f16) {
    return std::make_shared<ov::op::v0::Parameter>(et, shape);
}

std::shared_ptr<ov::op::v0::Constant> f16_const(const std::vector<float>& values, const ov::Shape& shape) {
    return ov::op::v0::Constant::create(ov::element::f16, shape, values);
}

/// Number of nodes of exactly type `T` (a derived gpu_opset op is not counted as its core base).
template <class T>
size_t count_exact(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        count += node->get_type_info() == T::get_type_info_static() ? 1 : 0;
    }
    return count;
}

template <class T>
std::shared_ptr<T> find_exact(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (node->get_type_info() == T::get_type_info_static()) {
            return ov::as_type_ptr<T>(node);
        }
    }
    return nullptr;
}

void run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<SparseAttentionFusion>();
    manager.run_passes(model);
}

struct ModelWithShapes {
    std::shared_ptr<ov::Model> model;
    std::vector<ov::PartialShape> result_shapes;
};

ModelWithShapes make_model(const ov::OutputVector& outputs) {
    ov::ParameterVector params;
    ov::ResultVector results;
    std::vector<ov::PartialShape> shapes;
    for (const auto& output : outputs) {
        results.push_back(std::make_shared<ov::op::v0::Result>(output));
        shapes.push_back(output.get_partial_shape());
    }
    // Every Parameter reachable from the results.
    ov::NodeVector stack;
    for (const auto& output : outputs) {
        stack.push_back(output.get_node_shared_ptr());
    }
    std::set<ov::Node*> visited;
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!visited.insert(node.get()).second) {
            continue;
        }
        if (auto p = ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
            params.push_back(p);
        }
        for (const auto& input : node->input_values()) {
            stack.push_back(input.get_node_shared_ptr());
        }
    }
    return {std::make_shared<ov::Model>(results, params), shapes};
}

void expect_result_shapes_compatible(const ModelWithShapes& m) {
    ASSERT_EQ(m.model->get_results().size(), m.result_shapes.size());
    for (size_t i = 0; i < m.result_shapes.size(); i++) {
        EXPECT_TRUE(m.model->get_results()[i]->get_input_partial_shape(0).compatible(m.result_shapes[i])) << "result " << i;
    }
}

// ---------------------------------------------------------------------------
// QSAPrepare + SparseAttentionIndexer (+ SparseAttention), non-paged
// ---------------------------------------------------------------------------

enum class query_form {
    full_rope,    // RoPE(RMS(x)) with rotary_ndims < Di on the whole head
    sliced_rope,  // Concat(RoPE(RMS(x)[..., :Dr]), RMS(x)[..., Dr:])
    bare          // an already prepared query: nothing to fuse in front of the indexer
};

constexpr float query_gain = 1.5f;

struct dense_options {
    ov::element::Type et = ov::element::f16;
    ov::element::Type key_length_et = ov::element::i64;
    bool index_weights = false;
    int64_t selection_heads = 1;
    bool same_key_length = true;
    bool empty_rope = false;
    query_form query = query_form::full_rope;
    bool query_blhd = true;       // RoPE output goes through the [N, L, H, D] -> [N, H, L, D] Transpose
    bool same_query_rope = true;  // the query rotates with the QSAPrepare cos/sin
    double query_eps = eps;
};

struct dense_graph {
    ModelWithShapes built;
    std::shared_ptr<ov::op::v17::QSAPrepare> prepare;
    std::shared_ptr<ov::op::v17::SparseAttentionIndexer> indexer;
    std::shared_ptr<ov::op::v0::Parameter> raw_query;
};

/// RMSNorm(raw, gain) -> RoPE -> [Transpose]: the index query as the public SparseAttentionIndexer gets it.
ov::Output<ov::Node> build_index_query(const dense_options& o,
                                       const ov::Output<ov::Node>& cos,
                                       const ov::Output<ov::Node>& sin,
                                       std::shared_ptr<ov::op::v0::Parameter>& raw) {
    const auto et = o.et;
    raw = param(o.query_blhd ? ov::PartialShape{1, -1, Hi, Di} : ov::PartialShape{1, Hi, -1, Di}, et);
    if (o.query == query_form::bare) {
        raw = param({1, Hi, -1, Di}, et);
        return raw;
    }
    auto gain = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, query_gain));
    auto norm = std::make_shared<ov::op::internal::RMS>(raw, gain, o.query_eps);

    ov::Output<ov::Node> rope_cos = cos, rope_sin = sin;
    if (!o.same_query_rope) {
        rope_cos = param({1, -1, Dr}, et);
        rope_sin = param({1, -1, Dr}, et);
    }
    const int64_t head_axis = o.query_blhd ? 2 : 1;
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {head_axis});
    auto cos_u = std::make_shared<ov::op::v0::Unsqueeze>(rope_cos, axis);
    auto sin_u = std::make_shared<ov::op::v0::Unsqueeze>(rope_sin, axis);

    ov::op::internal::RoPE::Config rope_cfg;
    rope_cfg.rotary_ndims = Dr;
    rope_cfg.cos_sin_ndims = Dr;
    rope_cfg.head_cnt = Hi;
    rope_cfg.head_size = Di;

    auto i64 = [](std::vector<int64_t> v) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
    };
    ov::Output<ov::Node> rotated;
    if (o.query == query_form::sliced_rope) {
        auto head = std::make_shared<ov::op::v8::Slice>(norm, i64({0}), i64({Dr}), i64({1}), i64({-1}));
        auto tail = std::make_shared<ov::op::v8::Slice>(norm, i64({Dr}), i64({std::numeric_limits<int64_t>::max()}), i64({1}), i64({-1}));
        auto rope = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{head, cos_u, sin_u}, rope_cfg);
        rotated = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rope, tail}, -1);
    } else {
        rotated = std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{norm, cos_u, sin_u}, rope_cfg);
    }
    if (!o.query_blhd) {
        return rotated;
    }
    return std::make_shared<ov::op::v1::Transpose>(rotated, i64({0, 2, 1, 3}));
}

dense_graph build_dense_indexer_graph(const dense_options& o = {}) {
    const auto et = o.et;
    auto raw_keys = param({1, -1, Di}, et);
    auto past_block_keys = param({1, -1, Di}, et);
    auto key_length = param({1}, o.key_length_et);
    ov::Output<ov::Node> rope_cos = param({1, -1, Dr}, et);
    ov::Output<ov::Node> rope_sin = param({1, -1, Dr}, et);
    ov::Output<ov::Node> past_rope = param({1, 2 * Dr}, et);
    if (o.empty_rope) {
        // rope_cos, rope_sin and past_incomplete_rope are disabled together.
        rope_cos = ov::op::v0::Constant::create(et, ov::Shape{0}, std::vector<float>{});
        rope_sin = ov::op::v0::Constant::create(et, ov::Shape{0}, std::vector<float>{});
        past_rope = ov::op::v0::Constant::create(et, ov::Shape{0}, std::vector<float>{});
    }
    auto k_norm = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, 1.0f));
    auto past_raw = param({1, r - 1, Di}, et);

    dense_graph g;
    ov::Output<ov::Node> index_query;
    if (o.empty_rope) {
        g.raw_query = param({1, Hi, -1, Di}, et);
        index_query = g.raw_query;
    } else {
        index_query = build_index_query(o, rope_cos, rope_sin, g.raw_query);
    }
    g.prepare = std::make_shared<ov::op::v17::QSAPrepare>(raw_keys, past_block_keys, key_length, rope_cos, rope_sin, k_norm, past_raw, past_rope, r, eps);

    ov::Output<ov::Node> indexer_length = key_length;
    if (!o.same_key_length) {
        indexer_length = param({1}, o.key_length_et);
    }
    ov::OutputVector indexer_inputs{index_query, g.prepare->output(0), indexer_length};
    if (o.index_weights) {
        indexer_inputs.push_back(param({1, -1, Hi}, et));
    }
    g.indexer = std::make_shared<ov::op::v17::SparseAttentionIndexer>(indexer_inputs, r, token_budget, o.selection_heads);

    g.built = make_model({g.indexer->output(0), g.indexer->output(1), g.prepare->output(0), g.prepare->output(1), g.prepare->output(2)});
    return g;
}

TEST(sparse_attention_fusion, qsa_prepare_and_indexer_fuse_into_qsa_indexer) {
    auto g = build_dense_indexer_graph();
    run_fusion(g.built.model);

    EXPECT_EQ(count_exact<ov::op::v17::QSAPrepare>(g.built.model), 0u);
    EXPECT_EQ(count_exact<ov::op::v17::SparseAttentionIndexer>(g.built.model), 0u);
    ASSERT_EQ(count_exact<op::QSAIndexer>(g.built.model), 1u);
    expect_result_shapes_compatible(g.built);

    auto fused = find_exact<op::QSAIndexer>(g.built.model);
    const auto& cfg = fused->get_config();
    EXPECT_EQ(cfg.key_block_size, static_cast<size_t>(r));
    EXPECT_EQ(cfg.block_topk, static_cast<size_t>(block_topk));
    EXPECT_EQ(cfg.num_selection_heads, 1u);
    EXPECT_EQ(cfg.indexer_rotary_dim, static_cast<size_t>(Dr));
    EXPECT_EQ(cfg.indexer_head_dim, static_cast<size_t>(Di));
    EXPECT_FLOAT_EQ(cfg.epsilon, eps);

    using In = op::QSAIndexer::InputIdx;
    // The kernels read the raw [B, L, H, D] query (norm and RoPE are applied inside) and an i32 [B] key_length.
    EXPECT_EQ(fused->get_input_node_shared_ptr(In::INDEX_QUERY), g.raw_query);
    EXPECT_EQ(fused->get_input_partial_shape(In::INDEX_QUERY), (ov::PartialShape{1, -1, Hi, Di}));
    auto q_norm = ov::as_type_ptr<ov::op::v0::Constant>(fused->get_input_node_shared_ptr(In::Q_NORM_WEIGHT));
    ASSERT_NE(q_norm, nullptr);
    EXPECT_EQ(q_norm->get_element_type(), ov::element::f16);
    EXPECT_EQ(q_norm->cast_vector<float>(), std::vector<float>(Di, query_gain));
    EXPECT_EQ(count_exact<ov::op::internal::RMS>(g.built.model), 0u) << "the query RMSNorm is fused";
    EXPECT_EQ(count_exact<ov::op::internal::RoPE>(g.built.model), 0u) << "the query RoPE is fused";
    EXPECT_EQ(fused->get_input_partial_shape(In::RAW_INDEX_KEYS), (ov::PartialShape{1, -1, 1, Di}));
    EXPECT_EQ(fused->get_input_element_type(In::KEY_LENGTH), ov::element::i32);
    EXPECT_EQ(fused->get_input_partial_shape(In::BLOCK_INDEX_KEYS_CACHE), (ov::PartialShape{1, 1, -1, Di}));
    EXPECT_EQ(fused->get_input_partial_shape(In::PAST_INCOMPLETE_RAW_K), (ov::PartialShape{1, 1, r - 1, Di}));
}

TEST(sparse_attention_fusion, qsa_indexer_fusion_accepts_i32_key_length) {
    dense_options o;
    o.key_length_et = ov::element::i32;
    auto g = build_dense_indexer_graph(o);
    run_fusion(g.built.model);
    EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 1u);
}

TEST(sparse_attention_fusion, qsa_indexer_fusion_fuses_partial_rotary_slices) {
    dense_options o;
    o.query = query_form::sliced_rope;
    auto g = build_dense_indexer_graph(o);
    run_fusion(g.built.model);

    ASSERT_EQ(count_exact<op::QSAIndexer>(g.built.model), 1u);
    EXPECT_EQ(find_exact<op::QSAIndexer>(g.built.model)->get_input_node_shared_ptr(op::QSAIndexer::InputIdx::INDEX_QUERY), g.raw_query);
    EXPECT_EQ(count_exact<ov::op::internal::RoPE>(g.built.model), 0u);
}

TEST(sparse_attention_fusion, qsa_indexer_fusion_transposes_heads_first_query) {
    dense_options o;
    o.query_blhd = false;  // RoPE already produces [N, H, L, D]
    auto g = build_dense_indexer_graph(o);
    run_fusion(g.built.model);

    ASSERT_EQ(count_exact<op::QSAIndexer>(g.built.model), 1u);
    auto fused = find_exact<op::QSAIndexer>(g.built.model);
    auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(fused->get_input_node_shared_ptr(op::QSAIndexer::InputIdx::INDEX_QUERY));
    ASSERT_NE(transpose, nullptr);
    EXPECT_EQ(transpose->get_input_node_shared_ptr(0), g.raw_query);
    EXPECT_EQ(fused->get_input_partial_shape(op::QSAIndexer::InputIdx::INDEX_QUERY), (ov::PartialShape{1, -1, Hi, Di}));
}

TEST(sparse_attention_fusion, qsa_indexer_fusion_bails_out_without_matching_query_side) {
    {
        dense_options o;
        o.query = query_form::bare;  // no RMSNorm/RoPE in front of the indexer
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "bare query";
        EXPECT_EQ(count_exact<ov::op::v17::QSAPrepare>(g.built.model), 1u);
    }
    {
        dense_options o;
        o.same_query_rope = false;  // the query must rotate with the block keys' cos/sin
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "different cos/sin";
    }
    {
        dense_options o;
        o.query_eps = 1e-3;  // one epsilon for the query norm and the block key norm
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "different epsilon";
    }
}

TEST(sparse_attention_fusion, qsa_indexer_fusion_bails_out_on_unsupported_configurations) {
    {
        dense_options o;
        o.index_weights = true;  // no kernel support for per-head weights
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "index_weights";
        EXPECT_EQ(count_exact<ov::op::v17::SparseAttentionIndexer>(g.built.model), 1u);
    }
    {
        dense_options o;
        o.selection_heads = 2;
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "num_selection_heads";
    }
    {
        dense_options o;
        o.et = ov::element::f32;  // the lowering of the QSA ops accepts f16 data only
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "f32";
    }
    {
        dense_options o;
        o.same_key_length = false;  // the kernel takes one key_length for both ops
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "different key_length";
    }
    {
        dense_options o;
        o.empty_rope = true;  // the kernels always rotate
        auto g = build_dense_indexer_graph(o);
        run_fusion(g.built.model);
        EXPECT_EQ(count_exact<op::QSAIndexer>(g.built.model), 0u) << "empty rope";
    }
}

// ---------------------------------------------------------------------------
// SparseAttention -> SparseSDPA
// ---------------------------------------------------------------------------

struct attention_options {
    size_t input_count = 7;  // 5: no key_length/scale, 6: + key_length, 7: + scale, 8: + sinks
    bool causal = true;
    int64_t query_block_size = 1;
    int64_t selection_heads = 1;
    ov::element::Type index_et = ov::element::i64;
};

ModelWithShapes build_attention_graph(const attention_options& o = {}) {
    auto query = param({1, Hq, -1, Dh});
    auto key = param({1, Hkv, -1, Dh});
    auto value = param({1, Hkv, -1, Dh});
    auto block_indices = param({1, o.selection_heads, -1, block_topk}, o.index_et);
    auto block_count = param({1, o.selection_heads, -1}, o.index_et);
    ov::OutputVector inputs{query, key, value, block_indices, block_count};
    if (o.input_count >= 6) {
        inputs.push_back(param({1}, ov::element::i64));
    }
    if (o.input_count >= 7) {
        inputs.push_back(f16_const({0.25f}, ov::Shape{}));
    }
    if (o.input_count >= 8) {
        inputs.push_back(param({Hq}));
    }
    auto attention = std::make_shared<ov::op::v17::SparseAttention>(inputs, r, o.query_block_size, o.causal);
    return make_model({attention->output(0)});
}

TEST(sparse_attention_fusion, sparse_attention_converts_to_sparse_sdpa) {
    auto m = build_attention_graph();
    run_fusion(m.model);

    EXPECT_EQ(count_exact<ov::op::v17::SparseAttention>(m.model), 0u);
    ASSERT_EQ(count_exact<op::SparseSDPA>(m.model), 1u);
    expect_result_shapes_compatible(m);

    auto sdpa = find_exact<op::SparseSDPA>(m.model);
    const auto& cfg = sdpa->get_config();
    EXPECT_EQ(cfg.num_heads, static_cast<size_t>(Hq));
    EXPECT_EQ(cfg.num_kv_heads, static_cast<size_t>(Hkv));
    EXPECT_EQ(cfg.num_selection_heads, 1u);
    EXPECT_EQ(cfg.k_head_size, static_cast<size_t>(Dh));
    EXPECT_EQ(cfg.v_head_size, static_cast<size_t>(Dh));
    EXPECT_EQ(cfg.key_block_size, static_cast<size_t>(r));
    EXPECT_TRUE(cfg.causal);

    using In = op::SparseSDPA::InputIdx;
    // The kernels read every integer input as int32.
    EXPECT_EQ(sdpa->get_input_element_type(In::SELECTED_BLOCK_INDICES), ov::element::i32);
    EXPECT_EQ(sdpa->get_input_element_type(In::SELECTED_BLOCK_COUNT), ov::element::i32);
    EXPECT_EQ(sdpa->get_input_element_type(In::KEY_LENGTH), ov::element::i32);
    ASSERT_TRUE(sdpa->has_scale_input());
    auto scale = ov::as_type_ptr<ov::op::v0::Constant>(sdpa->get_input_node_shared_ptr(In::SCALE));
    ASSERT_NE(scale, nullptr);
    EXPECT_FLOAT_EQ(scale->cast_vector<float>().at(0), 0.25f);
}

TEST(sparse_attention_fusion, sparse_attention_without_key_length_assumes_no_padding) {
    attention_options o;
    o.input_count = 5;
    o.index_et = ov::element::i32;
    auto m = build_attention_graph(o);
    run_fusion(m.model);

    ASSERT_EQ(count_exact<op::SparseSDPA>(m.model), 1u);
    auto sdpa = find_exact<op::SparseSDPA>(m.model);
    EXPECT_FALSE(sdpa->has_scale_input());
    auto key_length = ov::as_type_ptr<ov::op::v3::Broadcast>(sdpa->get_input_node_shared_ptr(op::SparseSDPA::InputIdx::KEY_LENGTH));
    EXPECT_NE(key_length, nullptr) << "key_length is synthesized from the key sequence length";
}

TEST(sparse_attention_fusion, sparse_attention_bails_out_on_unsupported_configurations) {
    {
        attention_options o;
        o.input_count = 8;  // attention sinks
        auto m = build_attention_graph(o);
        run_fusion(m.model);
        EXPECT_EQ(count_exact<op::SparseSDPA>(m.model), 0u) << "sinks";
        EXPECT_EQ(count_exact<ov::op::v17::SparseAttention>(m.model), 1u);
    }
    {
        attention_options o;
        o.causal = false;
        auto m = build_attention_graph(o);
        run_fusion(m.model);
        EXPECT_EQ(count_exact<op::SparseSDPA>(m.model), 0u) << "non-causal";
    }
    {
        attention_options o;
        o.query_block_size = 2;
        auto m = build_attention_graph(o);
        run_fusion(m.model);
        EXPECT_EQ(count_exact<op::SparseSDPA>(m.model), 0u) << "query_block_size";
    }
    {
        attention_options o;
        o.selection_heads = 2;
        auto m = build_attention_graph(o);
        run_fusion(m.model);
        EXPECT_EQ(count_exact<op::SparseSDPA>(m.model), 0u) << "num_selection_heads";
    }
}

// ---------------------------------------------------------------------------
// PagedQSAIndexer / PagedSparseAttention -> gpu_opset ops
// ---------------------------------------------------------------------------

ov::OutputVector make_paged_indexer_args() {
    return {param({-1, Hi * Di}),
            param({-1, Di}),
            param({-1, Dr}),
            param({-1, Dr}),
            param({Di}),
            param({Di}),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({pages, 1, page_size, Di}),
            param({pages, pages_blocks, 2 * Dr}),
            param({pages, 1, pages_blocks, Di})};
}

ov::op::internal::PagedQSAIndexer::Config make_paged_indexer_config() {
    ov::op::internal::PagedQSAIndexer::Config c;
    c.key_block_size = r;
    c.block_topk = block_topk;
    c.num_selection_heads = 1;
    c.indexer_rotary_dim = Dr;
    c.indexer_head_dim = Di;
    c.epsilon = eps;
    c.scale = 0.0f;
    return c;
}

TEST(sparse_attention_fusion, paged_qsa_indexer_converts_to_gpu_op_and_tags_caches) {
    using In = ov::op::internal::PagedQSAIndexer::InputIdx;
    auto args = make_paged_indexer_args();
    auto indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(args, make_paged_indexer_config());
    auto m = make_model({indexer->output(0), indexer->output(1)});
    run_fusion(m.model);

    EXPECT_EQ(count_exact<ov::op::internal::PagedQSAIndexer>(m.model), 0u);
    ASSERT_EQ(count_exact<op::PagedQSAIndexer>(m.model), 1u);
    expect_result_shapes_compatible(m);

    auto fused = find_exact<op::PagedQSAIndexer>(m.model);
    EXPECT_EQ(fused->get_config().block_topk, static_cast<size_t>(block_topk));
    for (size_t idx : {In::INDEX_KEY_CACHE, In::INDEX_BLOCK_ROPE_CACHE, In::BLOCK_INDEX_KEYS_CACHE}) {
        auto cache = ov::as_type_ptr<ov::op::v0::Parameter>(fused->get_input_node_shared_ptr(idx));
        ASSERT_NE(cache, nullptr);
        EXPECT_TRUE(cache->output(0).get_rt_info().count(in_place_cache_rt_key)) << "cache " << idx;
    }
    EXPECT_FALSE(fused->get_input_node_shared_ptr(In::PAST_LENS)->output(0).get_rt_info().count(in_place_cache_rt_key));

    // The produced gpu_opset op is not converted again.
    run_fusion(m.model);
    EXPECT_EQ(count_exact<op::PagedQSAIndexer>(m.model), 1u);
}

ov::OutputVector make_paged_attention_args(int64_t head_size = Dh) {
    return {param({-1, Hq * head_size}),
            param({-1, Hkv * head_size}),
            param({-1, Hkv * head_size}),
            param({pages, Hkv, page_size, head_size}),
            param({pages, Hkv, page_size, head_size}),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({-1}, ov::element::i32),
            param({-1, 1, block_topk}, ov::element::i32),
            param({-1, 1}, ov::element::i32),
            param({}),
            param({0})};
}

TEST(sparse_attention_fusion, paged_sparse_attention_converts_to_sparse_pa) {
    auto attention = std::make_shared<ov::op::internal::PagedSparseAttention>(make_paged_attention_args(), r);
    auto m = make_model({attention->output(0)});
    run_fusion(m.model);

    EXPECT_EQ(count_exact<ov::op::internal::PagedSparseAttention>(m.model), 0u);
    ASSERT_EQ(count_exact<op::SparsePA>(m.model), 1u);
    expect_result_shapes_compatible(m);

    const auto& cfg = find_exact<op::SparsePA>(m.model)->get_config();
    EXPECT_EQ(cfg.num_heads, static_cast<size_t>(Hq));
    EXPECT_EQ(cfg.num_kv_heads, static_cast<size_t>(Hkv));
    EXPECT_EQ(cfg.num_selection_heads, 1u);
    EXPECT_EQ(cfg.k_head_size, static_cast<size_t>(Dh));
    EXPECT_EQ(cfg.v_head_size, static_cast<size_t>(Dh));
    EXPECT_EQ(cfg.key_block_size, static_cast<size_t>(r));

    run_fusion(m.model);
    EXPECT_EQ(count_exact<op::SparsePA>(m.model), 1u);
}

TEST(sparse_attention_fusion, paged_sparse_attention_bails_out_on_unsupported_head_size) {
    auto attention = std::make_shared<ov::op::internal::PagedSparseAttention>(make_paged_attention_args(8), r);  // not a multiple of 16
    auto m = make_model({attention->output(0)});
    run_fusion(m.model);

    EXPECT_EQ(count_exact<op::SparsePA>(m.model), 0u);
    EXPECT_EQ(count_exact<ov::op::internal::PagedSparseAttention>(m.model), 1u);
}

}  // namespace
