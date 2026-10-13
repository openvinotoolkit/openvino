// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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
constexpr int64_t Hidx = 2;   // indexer heads
constexpr int64_t D = 32;     // hidden size
constexpr int64_t Hq = 2;     // main attention heads
constexpr int64_t Hkv = 2;    // main attention kv heads (Hq == Hkv here: no GQA needed to prove the pattern)
constexpr int64_t Dh = 16;    // main K/V head size
constexpr int64_t r = 4;      // compress ratio
constexpr int64_t block_topk = 1;

std::shared_ptr<ov::op::v0::Constant> i64_const(std::vector<int64_t> v) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
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

/// Common indexer-side sub-graph shared by the dense and paged builders below: a shared
/// `index_qk_proj`-style MatMul producing indexer Q and the current token's indexer K
/// contribution, RMSNorm gains, a shared packed rotary table, a `ReadValue`-backed raw-K
/// history, block-mean pooling with a static group size `r`, and the
/// `DenseSparseAttentionFusion`/`PagedSparseAttentionFusion`-recognized indexer score
/// signature (ReLU -> ReduceSum -> 1/sqrt(Di) -> TopK).
struct IndexerGraph {
    std::shared_ptr<ov::op::v0::Parameter> mixed;
    std::shared_ptr<ov::op::v11::TopK> topk;
    std::shared_ptr<ov::op::v6::ReadValue> raw_k_read_value;
};

IndexerGraph build_indexer_graph() {
    IndexerGraph g;
    g.mixed = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, D});

    auto qk_weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>((Hidx + 1) * Di), static_cast<size_t>(D)},
                                                  std::vector<float>((Hidx + 1) * Di * D, 0.01f));
    auto qk_proj = std::make_shared<ov::op::v0::MatMul>(g.mixed, qk_weight, false, true);  // [1,1,(Hidx+1)*Di]

    auto q_slice = std::make_shared<ov::op::v8::Slice>(qk_proj, i64_const({0, 0, 0}), i64_const({1, 1, Hidx * Di}), i64_const({1, 1, 1}));
    auto q_headsplit = std::make_shared<ov::op::v1::Reshape>(q_slice, i64_const({1, 1, Hidx, Di}), false);
    auto q_gain = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, 1.0f));
    auto q_normed = std::make_shared<ov::op::v1::Multiply>(q_headsplit, q_gain);

    auto k_slice = std::make_shared<ov::op::v8::Slice>(qk_proj, i64_const({0, 0, Hidx * Di}), i64_const({1, 1, (Hidx + 1) * Di}), i64_const({1, 1, 1}));
    auto k_headsplit = std::make_shared<ov::op::v1::Reshape>(k_slice, i64_const({1, 1, 1, Di}), false);
    auto k_gain = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(Di)}, std::vector<float>(Di, 1.0f));
    auto k_normed = std::make_shared<ov::op::v1::Multiply>(k_headsplit, k_gain);

    // A single already-packed [positions, Di] table shared by the Q and K legs.
    auto rope_table = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{64, static_cast<size_t>(Di)}, std::vector<float>(64 * Di, 0.0f));
    auto rope_slice = std::make_shared<ov::op::v8::Slice>(rope_table, i64_const({0, 0}), i64_const({1, Di}), i64_const({1, 1}));
    auto rope_reshaped = std::make_shared<ov::op::v1::Reshape>(rope_slice, i64_const({1, 1, 1, Di}), false);
    auto idx_q = std::make_shared<ov::op::v1::Add>(q_normed, rope_reshaped);
    auto new_k = std::make_shared<ov::op::v1::Add>(k_normed, rope_reshaped);

    // Indexer raw-K history: ReadValue -> Concat(new_k).
    auto raw_k_variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{1, ov::Dimension::dynamic(), Di}, ov::element::f32, "raw_k"});
    g.raw_k_read_value = std::make_shared<ov::op::v6::ReadValue>(raw_k_variable);
    auto new_k_flat = std::make_shared<ov::op::v1::Reshape>(new_k, i64_const({1, 1, Di}), false);
    auto raw_k_history = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{g.raw_k_read_value, new_k_flat}, 1);  // [1, ?, Di]

    // Block-mean pooling with a STATIC group size r (n_blocks stays dynamic).
    auto pool_reshaped = std::make_shared<ov::op::v1::Reshape>(raw_k_history, i64_const({1, -1, r, Di}), false);  // [1,?,r,Di]
    auto pooled = std::make_shared<ov::op::v1::ReduceMean>(pool_reshaped, i64_const({2}), false);                // [1,?,Di]
    auto idx_k = std::make_shared<ov::op::v0::Unsqueeze>(pooled, i64_const({1}));                                // [1,1,?,Di]

    // Indexer score signature: ReLU -> ReduceSum -> 1/sqrt(Di) -> TopK.
    auto logits = std::make_shared<ov::op::v0::MatMul>(idx_q, idx_k, false, true);  // [1,1,Hidx,?]
    auto relu = std::make_shared<ov::op::v0::Relu>(logits);
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(relu, i64_const({2}), false);  // [1,1,?]
    auto scaled =
        std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / std::sqrt(static_cast<float>(Di))}));
    g.topk = std::make_shared<ov::op::v11::TopK>(scaled, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {block_topk})->output(0), 2,
                                                 ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES);
    return g;
}

/// Threads `topk`'s output into `target` without constraining broadcast-compatibility
/// (any-rank ShapeOf -> scalar reduce -> * 0.0 -> Add), mirroring how the legacy
/// QSASDPAFusion test proved the downstream consumer is actually reachable from the
/// indexer's own selection.
ov::Output<ov::Node> zero_bias_from_topk(const ov::Output<ov::Node>& topk_out, const ov::Output<ov::Node>& target) {
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(topk_out, ov::element::i64);
    auto shape_f = std::make_shared<ov::op::v0::Convert>(shape, ov::element::f32);
    auto scalar = std::make_shared<ov::op::v1::ReduceSum>(shape_f, i64_const({0}), false);
    auto zeroed = std::make_shared<ov::op::v1::Multiply>(scalar, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
    return std::make_shared<ov::op::v1::Add>(target, zeroed);
}

constexpr int64_t s_cap = 8;  // main dense K/V cache capacity (static case)

struct DenseGraph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
};

DenseGraph build_dense_graph(bool static_kv_shape) {
    auto g = build_indexer_graph();

    ov::PartialShape kv_shape{1, Hkv, static_cast<int64_t>(static_kv_shape ? s_cap : -1), Dh};
    auto k_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, ov::element::f32, "main_k"});
    auto v_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, ov::element::f32, "main_v"});
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_variable);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_variable);
    auto k_biased = zero_bias_from_topk(g.topk->output(0), k_read);

    auto q_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, Hq, 1, Dh});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_param, k_biased, v_read, true);
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);

    DenseGraph out;
    out.sdpa = sdpa;
    out.model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{g.mixed, q_param});
    return out;
}

constexpr int64_t num_blocks = 2;
constexpr int64_t page_size = 16;

/// Builds a minimal, fully-valid 28-input `PagedAttentionExtension` call for one
/// sequence, one new token, past_lens=0 -- "disabled" optional inputs use the smallest
/// shape `validate_and_infer_types()` accepts for that rank (see paged_attention.cpp's
/// `input_check` table), not arbitrary placeholders.
struct PagedAttentionBuild {
    std::shared_ptr<ov::op::PagedAttentionExtension> op;
    std::shared_ptr<ov::op::v0::Parameter> key_cache;
    std::shared_ptr<ov::op::v0::Parameter> value_cache;
};

PagedAttentionBuild build_paged_attention(const ov::Output<ov::Node>& query, const ov::Output<ov::Node>& key, const ov::Output<ov::Node>& value) {
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{num_blocks, Hkv, page_size, Dh});
    auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{num_blocks, Hkv, page_size, Dh});
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
    args[10] = i32_scalar0;  // sliding_window (0 = unlimited)
    args[11] = f32_empty;    // alibi_slopes (empty = disabled)
    args[12] = i32_scalar0;  // max_context_len (0 = unlimited)
    args[13] = i32_scalar0;  // score_aggregation_window (0 = disabled)
    args[14] = i32_empty;    // rotated_block_indices (empty = disabled)
    args[15] = i32_empty;    // rotation_deltas (empty = disabled)
    args[16] = f32_empty;    // rotation_trig_lut (empty = disabled)
    args[17] = f32_empty;    // xattention_threshold (empty = disabled)
    args[18] = i32_scalar0;  // xattention_block_size (0 = disabled)
    args[19] = i32_scalar1;  // xattention_stride
    args[20] = f32_empty;    // sinks (empty = disabled)
    args[21] = i32_scalar0;  // adaptive_rkv_start_size (0 = no protection zone)
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
    std::shared_ptr<ov::op::PagedAttentionExtension> paged_attn;
};

PagedGraph build_paged_graph(bool include_paged_attention) {
    auto g = build_indexer_graph();

    auto q_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, Hq * Dh});
    auto k_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, Hkv * Dh});
    auto v_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, Hkv * Dh});

    PagedGraph out;
    ov::ParameterVector params{g.mixed, q_param, k_param, v_param};
    if (include_paged_attention) {
        auto k_biased = zero_bias_from_topk(g.topk->output(0), k_param);
        auto built = build_paged_attention(q_param, k_biased, v_param);
        out.paged_attn = built.op;
        params.push_back(built.key_cache);
        params.push_back(built.value_cache);
        auto result = std::make_shared<ov::op::v0::Result>(out.paged_attn->output(0));
        out.model = std::make_shared<ov::Model>(ov::ResultVector{result}, params);
    } else {
        // No reachable PagedAttentionExtension at all: the pass must leave this alone.
        auto result = std::make_shared<ov::op::v0::Result>(g.topk->output(0));
        out.model = std::make_shared<ov::Model>(ov::ResultVector{result}, params);
    }
    return out;
}

}  // namespace

TEST(sparse_attention_fusion, dense_bails_out_on_dynamic_kv_shape) {
    // Mirrors a real export with a dynamic sequence length: DenseSparseAttentionFusion
    // cannot bridge this case and must correctly leave the graph untouched.
    auto g = build_dense_graph(/*static_kv_shape=*/false);
    const size_t params_before = g.model->get_parameters().size();

    ov::pass::Manager manager;
    manager.register_pass<DenseSparseAttentionFusion>();
    manager.run_passes(g.model);

    EXPECT_EQ(count_ops<ov::op::internal::QSAIndexer>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::internal::SparseSDPA>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 1u);
    EXPECT_EQ(g.model->get_parameters().size(), params_before);
}

TEST(sparse_attention_fusion, dense_replaces_with_qsaindexer_and_sparsesdpa) {
    auto g = build_dense_graph(/*static_kv_shape=*/true);
    const size_t params_before = g.model->get_parameters().size();

    ov::pass::Manager manager;
    manager.register_pass<DenseSparseAttentionFusion>();
    manager.run_passes(g.model);

    ASSERT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 0u) << "the original SDPA must be replaced, not kept alongside";
    ASSERT_EQ(count_ops<ov::op::internal::QSAIndexer>(g.model), 1u);
    ASSERT_EQ(count_ops<ov::op::internal::SparseSDPA>(g.model), 1u);
    // The new cache tensors have no ReadValue/Assign counterpart in QSAIndexer's ABI:
    // three new Parameters must have been appended to the model's input signature.
    EXPECT_EQ(g.model->get_parameters().size(), params_before + 3);

    for (const auto& node : g.model->get_ops()) {
        if (auto indexer = ov::as_type_ptr<ov::op::internal::QSAIndexer>(node)) {
            const auto& config = indexer->get_config();
            EXPECT_EQ(config.compress_ratio, static_cast<size_t>(r));
            EXPECT_EQ(config.block_topk, static_cast<size_t>(block_topk));
            EXPECT_EQ(config.indexer_head_dim, static_cast<size_t>(Di));
        }
        if (auto sdpa = ov::as_type_ptr<ov::op::internal::SparseSDPA>(node)) {
            const auto& config = sdpa->get_config();
            EXPECT_EQ(config.num_heads, static_cast<size_t>(Hq));
            EXPECT_EQ(config.num_kv_heads, static_cast<size_t>(Hkv));
            EXPECT_EQ(config.k_head_size, static_cast<size_t>(Dh));
            EXPECT_EQ(config.sel_block_size, static_cast<size_t>(r));
        }
    }
}

TEST(sparse_attention_fusion, paged_bails_out_without_reachable_paged_attention) {
    auto g = build_paged_graph(/*include_paged_attention=*/false);
    const size_t params_before = g.model->get_parameters().size();

    ov::pass::Manager manager;
    manager.register_pass<PagedSparseAttentionFusion>();
    manager.run_passes(g.model);

    EXPECT_EQ(count_ops<ov::op::internal::PagedQSAIndexer>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::internal::SparsePA>(g.model), 0u);
    EXPECT_EQ(g.model->get_parameters().size(), params_before);
}

TEST(sparse_attention_fusion, paged_replaces_with_pagedqsaindexer_and_sparsepa) {
    auto g = build_paged_graph(/*include_paged_attention=*/true);
    const size_t params_before = g.model->get_parameters().size();

    ov::pass::Manager manager;
    manager.register_pass<PagedSparseAttentionFusion>();
    manager.run_passes(g.model);

    ASSERT_EQ(count_ops<ov::op::PagedAttentionExtension>(g.model), 0u) << "the original PagedAttentionExtension must be replaced, not kept alongside";
    ASSERT_EQ(count_ops<ov::op::internal::PagedQSAIndexer>(g.model), 1u);
    ASSERT_EQ(count_ops<ov::op::internal::SparsePA>(g.model), 1u);
    EXPECT_EQ(g.model->get_parameters().size(), params_before + 3);

    for (const auto& node : g.model->get_ops()) {
        if (auto indexer = ov::as_type_ptr<ov::op::internal::PagedQSAIndexer>(node)) {
            const auto& config = indexer->get_config();
            EXPECT_EQ(config.compress_ratio, static_cast<size_t>(r));
            EXPECT_EQ(config.block_topk, static_cast<size_t>(block_topk));
            EXPECT_EQ(config.indexer_head_dim, static_cast<size_t>(Di));
            EXPECT_EQ(config.page_size, static_cast<size_t>(page_size));
        }
        if (auto sparse_pa = ov::as_type_ptr<ov::op::internal::SparsePA>(node)) {
            const auto& config = sparse_pa->get_config();
            EXPECT_EQ(config.num_heads, static_cast<size_t>(Hq));
            EXPECT_EQ(config.num_kv_heads, static_cast<size_t>(Hkv));
            EXPECT_EQ(config.block_size, static_cast<size_t>(page_size));
            EXPECT_EQ(config.sel_block_size, static_cast<size_t>(r));
        }
    }
}
