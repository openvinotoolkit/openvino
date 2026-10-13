// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_fusion.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/sparse_pa.hpp"
#include "openvino/op/sparse_sdpa.hpp"

namespace ov::intel_gpu {

namespace {

/// Reads a constant that must hold exactly one value (through a Convert/Minimum wrapper).
ov::Output<ov::Node> resolve_constant_source(ov::Output<ov::Node> out) {
    for (int guard = 0; guard < 4; ++guard) {
        auto node = out.get_node_shared_ptr();
        if (ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            return out;
        }
        if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
            out = convert->input_value(0);
            continue;
        }
        if (auto minimum = ov::as_type_ptr<ov::op::v1::Minimum>(node)) {
            if (ov::as_type_ptr<ov::op::v0::Constant>(minimum->input_value(0).get_node_shared_ptr())) {
                out = minimum->input_value(0);
                continue;
            }
            if (ov::as_type_ptr<ov::op::v0::Constant>(minimum->input_value(1).get_node_shared_ptr())) {
                out = minimum->input_value(1);
                continue;
            }
        }
        break;
    }
    return out;
}

bool get_scalar_f32(const ov::Output<ov::Node>& out, float& value) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(out).get_node_shared_ptr());
    if (!c) {
        return false;
    }
    const auto v = c->cast_vector<float>();
    if (v.size() != 1) {
        return false;
    }
    value = v[0];
    return true;
}

bool close(float a, float b) {
    return std::fabs(a - b) <= 1e-5f * std::max(1.0f, std::fabs(b));
}

bool get_i64_vector(const ov::Output<ov::Node>& out, std::vector<int64_t>& value) {
    auto c = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(out).get_node_shared_ptr());
    if (!c) {
        return false;
    }
    value = c->cast_vector<int64_t>();
    return true;
}

size_t static_element_count(const ov::PartialShape& ps) {
    if (ps.is_dynamic()) {
        return 0;
    }
    size_t total = 1;
    for (const auto& d : ps) {
        total *= static_cast<size_t>(d.get_length());
    }
    return total;
}

/// Bounded BFS over data-flow ancestors of `start`, same discipline the legacy
/// QSAFusion/QSASDPAFusion used: prove facts by bounded structural search rather than a
/// single rigid pattern, so real (variably-shaped) RMSNorm/RoPE decompositions don't need
/// to be pinned down exactly.
template <typename T>
std::shared_ptr<T> find_ancestor(const ov::Output<ov::Node>& start, size_t max_depth, ov::Node* stop_at = nullptr) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (auto match = ov::as_type<T>(node)) {
            return std::static_pointer_cast<T>(match->shared_from_this());
        }
        if (node == stop_at || depth >= max_depth) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            queue.push_back({node->get_input_node_ptr(i), depth + 1});
        }
    }
    return nullptr;
}

template <typename T>
std::shared_ptr<T> find_descendant(const ov::Output<ov::Node>& start, size_t max_depth) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (auto match = ov::as_type<T>(node)) {
            return std::static_pointer_cast<T>(match->shared_from_this());
        }
        if (depth >= max_depth) {
            continue;
        }
        for (auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                queue.push_back({input.get_node(), depth + 1});
            }
        }
    }
    return nullptr;
}

std::shared_ptr<ov::Node> find_ancestor_if(const ov::Output<ov::Node>& start,
                                            size_t max_depth,
                                            ov::Node* stop_at,
                                            const std::function<bool(ov::Node*)>& pred) {
    std::unordered_set<ov::Node*> visited;
    std::deque<std::pair<ov::Node*, size_t>> queue;
    queue.push_back({start.get_node(), 0});
    while (!queue.empty()) {
        auto [node, depth] = queue.front();
        queue.pop_front();
        if (!node || !visited.insert(node).second) {
            continue;
        }
        if (pred(node)) {
            return node->shared_from_this();
        }
        if (node == stop_at || depth >= max_depth) {
            continue;
        }
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            queue.push_back({node->get_input_node_ptr(i), depth + 1});
        }
    }
    return nullptr;
}

/// The QSA indexer score pattern: scores = ReduceSum(ReLU(MatMul(q_idx, k_block,
/// transpose_b)), axis=head) * 1/sqrt(Di); topk = TopK(scores, block_topk). Tolerates
/// the reshape-before-ReLU / divide-instead-of-multiply / mask-before-TopK variants real
/// checkpoint exports use.
struct IndexerPattern {
    std::shared_ptr<ov::Node> logits;
    std::shared_ptr<ov::Node> reduced;
    std::shared_ptr<ov::Node> scaled_mul;
    std::shared_ptr<ov::Node> scaled_div;
    std::shared_ptr<ov::Node> topk;
};

IndexerPattern build_indexer_pattern() {
    using namespace ov::pass::pattern;
    IndexerPattern p;
    p.logits = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto logits_reshaped = wrap_type<ov::op::v1::Reshape>({p.logits, any_input()});
    auto relu_input = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{p.logits, logits_reshaped});
    auto relu = wrap_type<ov::op::v0::Relu>({relu_input});
    p.reduced = wrap_type<ov::op::v1::ReduceSum>({relu, any_input()});
    p.scaled_mul = wrap_type<ov::op::v1::Multiply>({p.reduced, any_input()});
    p.scaled_div = wrap_type<ov::op::v1::Divide>({p.reduced, any_input()});
    auto scaled = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{p.scaled_mul, p.scaled_div});
    auto masked = wrap_type<ov::op::v1::Select>({any_input(), any_input(), scaled});
    auto topk_input = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{scaled, masked});
    p.topk = wrap_type<ov::op::v11::TopK>({topk_input, any_input()});
    return p;
}

/// Proven facts extracted from a matched `IndexerPattern`.
struct IndexerSignature {
    std::shared_ptr<ov::op::v0::MatMul> logits_mm;
    size_t indexer_head_dim = 0;
    int64_t block_topk = 0;
};

bool verify_indexer_signature(const ov::pass::pattern::PatternValueMap& pm,
                               const IndexerPattern& p,
                               const std::shared_ptr<ov::Node>& root,
                               IndexerSignature& out) {
    auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(pm.at(p.logits).get_node_shared_ptr());
    if (!mm) {
        return false;
    }

    const auto& q_ps = mm->input_value(0).get_partial_shape();
    if (q_ps.rank().is_dynamic() || q_ps[q_ps.size() - 1].is_dynamic()) {
        return false;
    }
    const auto Di = static_cast<size_t>(q_ps[q_ps.size() - 1].get_length());
    const float expected_mul_scale = 1.0f / std::sqrt(static_cast<float>(Di));

    float scale = 0.0f;
    if (pm.count(p.scaled_mul)) {
        if (!get_scalar_f32(pm.at(p.scaled_mul).get_node_shared_ptr()->input_value(1), scale) || !close(scale, expected_mul_scale)) {
            return false;
        }
    } else if (pm.count(p.scaled_div)) {
        if (!get_scalar_f32(pm.at(p.scaled_div).get_node_shared_ptr()->input_value(1), scale) || !close(scale, 1.0f / expected_mul_scale)) {
            return false;
        }
    } else {
        return false;
    }

    std::vector<int64_t> axes;
    if (!get_i64_vector(pm.at(p.reduced).get_node_shared_ptr()->input_value(1), axes) || axes.size() != 1) {
        return false;
    }

    std::vector<int64_t> k_value;
    if (!get_i64_vector(root->input_value(1), k_value) || k_value.size() != 1 || k_value[0] <= 0) {
        return false;
    }

    out.logits_mm = mm;
    out.indexer_head_dim = Di;
    out.block_topk = k_value[0];
    return true;
}

/// Facts shared by both the paged and dense rewrites once the indexer score signature
/// is proven: the shared QK projection, per-leg RMSNorm gains, and the RoPE table.
struct SharedIndexerFacts {
    std::shared_ptr<ov::op::v0::MatMul> qk_proj_mm;
    ov::Output<ov::Node> qk_weight_src;
    std::shared_ptr<ov::op::v0::Constant> q_norm_const;
    std::shared_ptr<ov::op::v0::Constant> k_norm_const;
    std::shared_ptr<ov::op::v0::Constant> rope_const;
    size_t Hidx = 0;
    size_t D = 0;
    size_t indexer_rotary_dim = 0;
};

/// Proves the shared `[Hidx*Di + Di, D]` INDEXER_QK_WEIGHT projection convention, the
/// per-leg RMSNorm gain constants, and a single already-packed RoPE table, exactly as
/// the legacy QSASDPAFusion did. `idx_q`/`idx_k` are the (post-norm/RoPE) operands of the
/// indexer score MatMul.
bool prove_shared_indexer_facts(const ov::Output<ov::Node>& idx_q, const ov::Output<ov::Node>& idx_k, size_t Di, SharedIndexerFacts& out) {
    auto idx_q_proj_mm = find_ancestor<ov::op::v0::MatMul>(idx_q, 40);
    auto idx_k_proj_mm = find_ancestor<ov::op::v0::MatMul>(idx_k, 40);
    if (!idx_q_proj_mm || idx_q_proj_mm != idx_k_proj_mm || !idx_q_proj_mm->get_transpose_b()) {
        return false;
    }
    auto qk_weight_src = resolve_constant_source(idx_q_proj_mm->input_value(1));
    auto qk_weight_const = ov::as_type_ptr<ov::op::v0::Constant>(qk_weight_src.get_node_shared_ptr());
    if (!qk_weight_const) {
        return false;
    }
    const auto& qk_weight_ps = qk_weight_const->get_output_partial_shape(0);
    if (qk_weight_ps.rank().is_dynamic() || qk_weight_ps.size() != 2 || qk_weight_ps.is_dynamic()) {
        return false;
    }
    const auto qk_out_features = static_cast<size_t>(qk_weight_ps[0].get_length());
    const auto D = static_cast<size_t>(qk_weight_ps[1].get_length());
    if (Di == 0 || qk_out_features % Di != 0) {
        return false;
    }
    const size_t Hidx = qk_out_features / Di - 1;
    if (Hidx == 0) {
        return false;
    }

    auto find_norm_gain = [&](const ov::Output<ov::Node>& leg) {
        return ov::as_type_ptr<ov::op::v0::Constant>(find_ancestor_if(leg, 25, idx_q_proj_mm.get(), [Di](ov::Node* n) {
            auto c = ov::as_type<ov::op::v0::Constant>(n);
            return c && static_element_count(c->get_output_partial_shape(0)) == Di;
        }));
    };
    auto q_norm_const = find_norm_gain(idx_q);
    auto k_norm_const = find_norm_gain(idx_k);
    if (!q_norm_const || !k_norm_const) {
        return false;
    }

    auto rope_node = find_ancestor_if(idx_q, 25, idx_q_proj_mm.get(), [&](ov::Node* n) {
        auto c = ov::as_type<ov::op::v0::Constant>(n);
        if (!c || c == qk_weight_const.get() || c == q_norm_const.get()) {
            return false;
        }
        const auto& ps = c->get_output_partial_shape(0);
        if (ps.rank().is_dynamic() || ps.size() != 2 || ps.is_dynamic()) {
            return false;
        }
        const auto width = static_cast<size_t>(ps[1].get_length());
        return width >= 2 && width <= 2 * Di && width < D;
    });
    auto rope_const = ov::as_type_ptr<ov::op::v0::Constant>(rope_node);
    if (!rope_const) {
        return false;
    }

    out.qk_proj_mm = idx_q_proj_mm;
    out.qk_weight_src = qk_weight_src;
    out.q_norm_const = q_norm_const;
    out.k_norm_const = k_norm_const;
    out.rope_const = rope_const;
    out.Hidx = Hidx;
    out.D = D;
    out.indexer_rotary_dim = static_cast<size_t>(rope_const->get_output_partial_shape(0)[1].get_length());
    return true;
}

/// Rebuilds the pre-norm, pre-RoPE [1, 1, Hidx, Di] / [1, 1, 1, Di] indexer query/key
/// views directly from the shared projection's own output, rather than searching the
/// matched graph for them: QSAIndexer/PagedQSAIndexer apply RMSNorm + RoPE internally,
/// so feeding them an already-normalized/rotated value would double-apply both.
void split_qk_projection(const std::shared_ptr<ov::op::v0::MatMul>& qk_proj_mm,
                          size_t Hidx,
                          size_t Di,
                          ov::NodeVector& new_ops,
                          ov::Output<ov::Node>& q_idx,
                          ov::Output<ov::Node>& k_idx) {
    auto i64c = [&](std::vector<int64_t> v) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
        new_ops.push_back(c);
        return c;
    };
    const auto hidx_di = static_cast<int64_t>(Hidx * Di);
    auto q_slice = std::make_shared<ov::op::v8::Slice>(qk_proj_mm, i64c({0, 0, 0}), i64c({1, 1, hidx_di}), i64c({1, 1, 1}));
    new_ops.push_back(q_slice);
    auto q_reshaped = std::make_shared<ov::op::v1::Reshape>(q_slice, i64c({1, 1, static_cast<int64_t>(Hidx), static_cast<int64_t>(Di)}), false);
    new_ops.push_back(q_reshaped);
    auto k_slice = std::make_shared<ov::op::v8::Slice>(qk_proj_mm, i64c({0, 0, hidx_di}), i64c({1, 1, hidx_di + static_cast<int64_t>(Di)}), i64c({1, 1, 1}));
    new_ops.push_back(k_slice);
    auto k_reshaped = std::make_shared<ov::op::v1::Reshape>(k_slice, i64c({1, 1, 1, static_cast<int64_t>(Di)}), false);
    new_ops.push_back(k_reshaped);
    q_idx = q_reshaped->output(0);
    k_idx = k_reshaped->output(0);
}

}  // namespace

bool DenseSparseAttentionFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto pattern = build_indexer_pattern();
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(pattern.topk, "DenseSparseAttentionFusion");

    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto root = ov::as_type_ptr<ov::op::v11::TopK>(node);
        if (!root || !matcher->match(root->output(0))) {
            continue;
        }
        const auto pm = matcher->get_pattern_value_map();

        IndexerSignature sig;
        if (!verify_indexer_signature(pm, pattern, root, sig)) {
            continue;
        }
        const size_t Di = sig.indexer_head_dim;
        const auto idx_q = sig.logits_mm->input_value(0);
        const auto idx_k = sig.logits_mm->input_value(1);

        // The indexer K leg must carry its own persistent (ReadValue-backed) raw-K
        // history: this is what distinguishes a real incremental/stateful graph from a
        // one-shot reference computation, and it is the only concretely-validated
        // decomposed source pattern this pass can soundly bridge (see this pass's own
        // class doc comment for why QSAIndexer's caller-owned cache ABI cannot
        // participate in the ordinary ReadValue/Assign idiom).
        auto raw_k_read_value = find_ancestor<ov::op::v6::ReadValue>(idx_k, 60);
        if (!raw_k_read_value) {
            continue;
        }

        SharedIndexerFacts facts;
        if (!prove_shared_indexer_facts(idx_q, idx_k, Di, facts)) {
            continue;
        }

        auto sdpa = find_descendant<ov::op::v13::ScaledDotProductAttention>(root->output(0), 100);
        if (!sdpa) {
            sdpa = find_descendant<ov::op::v13::ScaledDotProductAttention>(root->output(1), 100);
        }
        if (!sdpa) {
            continue;
        }
        const auto query_in = sdpa->input_value(0);
        const auto key_in = sdpa->input_value(1);
        const auto value_in = sdpa->input_value(2);

        auto key_read_value = find_ancestor<ov::op::v6::ReadValue>(key_in, 40);
        auto value_read_value = find_ancestor<ov::op::v6::ReadValue>(value_in, 40);
        if (!key_read_value || !value_read_value || key_read_value == raw_k_read_value || value_read_value == raw_k_read_value) {
            continue;
        }

        const auto& q_ps = query_in.get_partial_shape();
        const auto& k_ps = key_in.get_partial_shape();
        const auto& v_ps = value_in.get_partial_shape();
        if (q_ps.rank().is_dynamic() || q_ps.size() != 4 || k_ps.rank().is_dynamic() || k_ps.size() != 4 || v_ps.rank().is_dynamic() ||
            v_ps.size() != 4 || q_ps.is_dynamic() || k_ps.is_dynamic() || v_ps.is_dynamic() || q_ps[0].get_length() != 1) {
            continue;
        }
        const size_t Hq = static_cast<size_t>(q_ps[1].get_length());
        const size_t Dh = static_cast<size_t>(q_ps[3].get_length());
        const size_t Hkv = static_cast<size_t>(k_ps[1].get_length());
        const size_t S = static_cast<size_t>(k_ps[2].get_length());
        const size_t Dv = static_cast<size_t>(v_ps[3].get_length());
        if (k_ps[0].get_length() != 1 || v_ps[0].get_length() != 1 || v_ps[2].get_length() != static_cast<int64_t>(S)) {
            continue;
        }

        // compress_ratio `r`: the indexer's own from-scratch block-mean pooling.
        auto pool = find_descendant<ov::op::v1::ReduceMean>(raw_k_read_value->output(0), 60);
        if (!pool) {
            continue;
        }
        std::vector<int64_t> pool_axes;
        if (!get_i64_vector(pool->input_value(1), pool_axes) || pool_axes.size() != 1) {
            continue;
        }
        const auto& pool_in_ps = pool->input_value(0).get_partial_shape();
        if (pool_in_ps.rank().is_dynamic()) {
            continue;
        }
        const auto pool_rank = static_cast<int64_t>(pool_in_ps.size());
        const int64_t pool_axis = pool_axes[0] < 0 ? pool_axes[0] + pool_rank : pool_axes[0];
        if (pool_axis < 0 || pool_axis >= pool_rank || pool_in_ps[pool_axis].is_dynamic()) {
            continue;
        }
        const size_t r = static_cast<size_t>(pool_in_ps[pool_axis].get_length());
        if (r == 0 || S % r != 0) {
            continue;
        }

        ov::op::internal::QSAIndexer::Config indexer_config{};
        indexer_config.compress_ratio = r;
        indexer_config.block_topk = static_cast<size_t>(sig.block_topk);
        indexer_config.h_sel = 1;
        indexer_config.indexer_rotary_dim = facts.indexer_rotary_dim;
        indexer_config.indexer_head_dim = Di;
        indexer_config.eps = 1e-6f;
        indexer_config.scale = 1.0f / std::sqrt(static_cast<float>(Di));
        std::string reason;
        if (!ov::op::internal::QSAIndexer::validate_config(indexer_config, reason)) {
            continue;
        }
        ov::op::internal::SparseSDPA::Config attn_config{};
        attn_config.num_heads = Hq;
        attn_config.num_kv_heads = Hkv;
        attn_config.h_sel = 1;
        attn_config.k_head_size = Dh;
        attn_config.v_head_size = Dv;
        attn_config.sel_block_size = r;
        attn_config.q_block_size = 1;
        attn_config.causal = true;
        if (!ov::op::internal::SparseSDPA::validate_config(attn_config, reason)) {
            continue;
        }

        // --- everything required is proven: build the replacement sub-graph --------
        ov::NodeVector new_ops;
        auto i64c = [&](std::vector<int64_t> v) {
            auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
            new_ops.push_back(c);
            return c;
        };

        ov::Output<ov::Node> q_idx_in, k_idx_in;
        split_qk_projection(facts.qk_proj_mm, facts.Hidx, Di, new_ops, q_idx_in, k_idx_in);

        // PAST_LENS: length of the indexer's own raw-K history before this call.
        auto raw_k_shape = std::make_shared<ov::op::v3::ShapeOf>(raw_k_read_value->output(0), ov::element::i64);
        new_ops.push_back(raw_k_shape);
        auto past_lens = std::make_shared<ov::op::v8::Gather>(raw_k_shape, i64c({1}), i64c({0}));
        new_ops.push_back(past_lens);
        auto past_len_scalar = std::make_shared<ov::op::v0::Squeeze>(past_lens, i64c({0}));
        new_ops.push_back(past_len_scalar);

        // POSITION_IDS: range(past_len, past_len + L) -- L is q_idx_in's own L axis.
        auto qk_shape = std::make_shared<ov::op::v3::ShapeOf>(facts.qk_proj_mm->output(0), ov::element::i64);
        new_ops.push_back(qk_shape);
        auto l_dim = std::make_shared<ov::op::v8::Gather>(qk_shape, i64c({1}), i64c({0}));
        new_ops.push_back(l_dim);
        auto l_scalar = std::make_shared<ov::op::v0::Squeeze>(l_dim, i64c({0}));
        new_ops.push_back(l_scalar);
        auto position_stop = std::make_shared<ov::op::v1::Add>(past_len_scalar, l_scalar);
        new_ops.push_back(position_stop);
        auto range_step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        new_ops.push_back(range_step);
        auto position_ids = std::make_shared<ov::op::v4::Range>(past_len_scalar, position_stop, range_step, ov::element::i64);
        new_ops.push_back(position_ids);

        // Indexer cache tensors: QSAIndexer's ABI is caller-owned/in-place with no
        // ReadValue counterpart (see class doc comment) -- new Parameters are the only
        // sound bridge from this ReadValue-based source pattern. The indexer's own raw-K
        // capacity is assumed equal to the main attention's dense K/V capacity S, mirroring
        // the legacy fused op's identical assumption (pa_block_size doubled as both).
        auto raw_k_cache = std::make_shared<ov::op::v0::Parameter>(
            raw_k_read_value->get_output_element_type(0), ov::PartialShape{1, 1, static_cast<int64_t>(S), static_cast<int64_t>(Di)});
        raw_k_cache->set_friendly_name(raw_k_read_value->get_variable_id() + "_raw_k_cache");
        auto summary_cache = std::make_shared<ov::op::v0::Parameter>(
            raw_k_read_value->get_output_element_type(0), ov::PartialShape{1, 1, static_cast<int64_t>(S / r), static_cast<int64_t>(Di)});
        summary_cache->set_friendly_name(raw_k_read_value->get_variable_id() + "_summary_cache");
        auto position_cache =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, static_cast<int64_t>(S / r)});
        position_cache->set_friendly_name(raw_k_read_value->get_variable_id() + "_position_cache");
        model->add_parameters({raw_k_cache, summary_cache, position_cache});

        ov::OutputVector indexer_args(ov::op::internal::QSAIndexer::InputIdx::COUNT);
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::Q_IDX] = q_idx_in;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::K_IDX] = k_idx_in;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::POSITION_IDS] = position_ids;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::ROTARY_COS_SIN] = facts.rope_const;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::SUMMARY_CACHE] = summary_cache;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::INDEXER_RAW_K_CACHE] = raw_k_cache;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::Q_NORM_WEIGHT] = facts.q_norm_const;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::K_NORM_WEIGHT] = facts.k_norm_const;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::PAST_LENS] = past_lens;
        indexer_args[ov::op::internal::QSAIndexer::InputIdx::INDEXER_POSITION_CACHE] = position_cache;
        auto qsa_indexer = std::make_shared<ov::op::internal::QSAIndexer>(indexer_args, indexer_config);
        new_ops.push_back(qsa_indexer);

        ov::OutputVector attn_args(ov::op::internal::SparseSDPA::InputIdx::COUNT_MIN);
        attn_args[ov::op::internal::SparseSDPA::InputIdx::QUERY] = query_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::KEY] = key_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::VALUE] = value_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::SEL_INDICES] = qsa_indexer->output(0);
        attn_args[ov::op::internal::SparseSDPA::InputIdx::SEL_COUNT] = qsa_indexer->output(1);
        auto sparse_sdpa = std::make_shared<ov::op::internal::SparseSDPA>(attn_args, attn_config, sdpa->get_output_element_type(0));
        new_ops.push_back(sparse_sdpa);

        ov::copy_runtime_info(ov::NodeVector{root, sdpa}, new_ops);
        ov::replace_node(sdpa, sparse_sdpa);

        // Retire any Assign feeding the raw-K Variable: QSAIndexer mutates its cache
        // Parameter in place and returns no updated history, so there is no value left
        // to feed it. The Variable declaration itself is left in place rather than
        // removed -- harmless if unused, and safer than asserting nothing else in the
        // graph still structurally touches it.
        auto variable = raw_k_read_value->get_variable();
        for (const auto& sink : model->get_sinks()) {
            if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(sink)) {
                if (assign->get_variable_id() == variable->get_info().variable_id) {
                    model->remove_sink(sink);
                }
            }
        }
        changed = true;
    }
    return changed;
}

bool PagedSparseAttentionFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto pattern = build_indexer_pattern();
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(pattern.topk, "PagedSparseAttentionFusion");

    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        auto root = ov::as_type_ptr<ov::op::v11::TopK>(node);
        if (!root || !matcher->match(root->output(0))) {
            continue;
        }
        const auto pm = matcher->get_pattern_value_map();

        IndexerSignature sig;
        if (!verify_indexer_signature(pm, pattern, root, sig)) {
            continue;
        }
        const size_t Di = sig.indexer_head_dim;
        const auto idx_q = sig.logits_mm->input_value(0);
        const auto idx_k = sig.logits_mm->input_value(1);

        SharedIndexerFacts facts;
        if (!prove_shared_indexer_facts(idx_q, idx_k, Di, facts)) {
            continue;
        }

        // Unlike the dense case, a paged/continuous-batching source graph is not
        // expected to use ReadValue at all: `PagedAttentionExtension` already carries the
        // page table as plain tensor inputs. This is the only concretely-grounded
        // paged anchor available in this codebase (no validated worked example exists
        // for this path -- see this pass's own class doc comment); everything else about
        // the indexer's own cache lifecycle is, same as the dense case, bridged via new
        // Parameters rather than guessed from an unproven decomposed pattern.
        auto paged_attn = find_descendant<ov::op::PagedAttentionExtension>(root->output(0), 100);
        if (!paged_attn) {
            paged_attn = find_descendant<ov::op::PagedAttentionExtension>(root->output(1), 100);
        }
        if (!paged_attn) {
            continue;
        }
        // PagedAttentionExtension has 3 outputs (query result, scores, diversity); SparsePA
        // only produces one. Outputs 1/2 are advanced/optional features (score
        // aggregation, adaptive-RKV diversity) this pass does not attempt to reproduce --
        // bail out rather than silently drop a real consumer of either.
        if (!paged_attn->output(1).get_target_inputs().empty() || !paged_attn->output(2).get_target_inputs().empty()) {
            continue;
        }

        const auto query_in = paged_attn->input_value(0);
        const auto key_in = paged_attn->input_value(1);
        const auto value_in = paged_attn->input_value(2);
        const auto key_cache_in = paged_attn->input_value(3);
        const auto value_cache_in = paged_attn->input_value(4);
        const auto past_lens_in = paged_attn->input_value(5);
        const auto subsequence_begins_in = paged_attn->input_value(6);
        const auto block_indices_in = paged_attn->input_value(7);
        const auto block_indices_begins_in = paged_attn->input_value(8);
        const auto scale_in = paged_attn->input_value(9);

        const auto& key_cache_ps = key_cache_in.get_partial_shape();
        if (key_cache_ps.rank().is_dynamic() || key_cache_ps.size() != 4 || key_cache_ps.is_dynamic()) {
            continue;
        }
        const size_t num_blocks = static_cast<size_t>(key_cache_ps[0].get_length());
        const size_t Hkv = static_cast<size_t>(key_cache_ps[1].get_length());
        const size_t page_size = static_cast<size_t>(key_cache_ps[2].get_length());
        const size_t Dh = static_cast<size_t>(key_cache_ps[3].get_length());
        const auto& value_cache_ps = value_cache_in.get_partial_shape();
        if (value_cache_ps.rank().is_dynamic() || value_cache_ps.size() != 4 || value_cache_ps.is_dynamic()) {
            continue;
        }
        const size_t Dv = static_cast<size_t>(value_cache_ps[3].get_length());
        const auto& query_ps = query_in.get_partial_shape();
        if (query_ps.rank().is_dynamic() || query_ps.size() != 2 || query_ps.is_dynamic()) {
            continue;
        }
        const size_t Hq_Dh = static_cast<size_t>(query_ps[1].get_length());
        if (Dh == 0 || Hq_Dh % Dh != 0) {
            continue;
        }
        const size_t Hq = Hq_Dh / Dh;

        // compress_ratio `r`: the indexer's own from-scratch block-mean pooling (same
        // proof as the dense case, just walked backward from idx_k since there is no
        // ReadValue anchor to search forward from in a paged/continuous-batching graph).
        auto pool = find_ancestor<ov::op::v1::ReduceMean>(idx_k, 60);
        if (!pool) {
            continue;
        }
        std::vector<int64_t> pool_axes;
        if (!get_i64_vector(pool->input_value(1), pool_axes) || pool_axes.size() != 1) {
            continue;
        }
        const auto& pool_in_ps = pool->input_value(0).get_partial_shape();
        if (pool_in_ps.rank().is_dynamic()) {
            continue;
        }
        const auto pool_rank = static_cast<int64_t>(pool_in_ps.size());
        const int64_t pool_axis = pool_axes[0] < 0 ? pool_axes[0] + pool_rank : pool_axes[0];
        if (pool_axis < 0 || pool_axis >= pool_rank || pool_in_ps[pool_axis].is_dynamic()) {
            continue;
        }
        const size_t r = static_cast<size_t>(pool_in_ps[pool_axis].get_length());
        if (r == 0 || page_size % r != 0) {
            continue;
        }

        ov::op::internal::PagedQSAIndexer::Config indexer_config{};
        indexer_config.compress_ratio = r;
        indexer_config.block_topk = static_cast<size_t>(sig.block_topk);
        indexer_config.h_sel = 1;
        indexer_config.indexer_rotary_dim = facts.indexer_rotary_dim;
        indexer_config.indexer_head_dim = Di;
        indexer_config.page_size = page_size;
        indexer_config.eps = 1e-6f;
        indexer_config.scale = 1.0f / std::sqrt(static_cast<float>(Di));
        std::string reason;
        if (!ov::op::internal::PagedQSAIndexer::validate_config(indexer_config, reason)) {
            continue;
        }
        ov::op::internal::SparsePA::Config attn_config{};
        attn_config.num_heads = Hq;
        attn_config.num_kv_heads = Hkv;
        attn_config.h_sel = 1;
        attn_config.k_head_size = Dh;
        attn_config.v_head_size = Dv;
        attn_config.block_size = page_size;
        attn_config.sel_block_size = r;
        attn_config.q_block_size = 1;
        attn_config.causal = true;
        if (get_scalar_f32(scale_in, attn_config.scale)) {
            // proven below via validate_config; fall through either way
        }
        if (!ov::op::internal::SparsePA::validate_config(attn_config, reason)) {
            continue;
        }

        ov::NodeVector new_ops;
        auto i64c = [&](std::vector<int64_t> v) {
            auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
            new_ops.push_back(c);
            return c;
        };

        ov::Output<ov::Node> q_idx_in, k_idx_in;
        split_qk_projection(facts.qk_proj_mm, facts.Hidx, Di, new_ops, q_idx_in, k_idx_in);
        // Flatten the [1,1,Hidx,Di]/[1,1,1,Di] views down to PagedQSAIndexer's token-major
        // [tokens, Hidx*Di]/[tokens, Di] convention (tokens == 1 along this call's L axis
        // for the proof above, generalized the same way query_in already is).
        auto flatten_tokens = [&](const ov::Output<ov::Node>& v, size_t heads) {
            auto reshaped = std::make_shared<ov::op::v1::Reshape>(v, i64c({-1, static_cast<int64_t>(heads * Di)}), true);
            new_ops.push_back(reshaped);
            return reshaped->output(0);
        };
        auto q_idx_flat = flatten_tokens(q_idx_in, facts.Hidx);
        auto k_idx_flat = flatten_tokens(k_idx_in, 1);

        // POSITION_IDS: derived from subsequence_begins the same way the main
        // PagedAttentionExtension call already does (per-token absolute position).
        auto position_ids = std::make_shared<ov::op::v8::Gather>(subsequence_begins_in, i64c({0}), i64c({0}));
        new_ops.push_back(position_ids);

        const size_t summaries_per_page = page_size / r;
        auto raw_k_cache = std::make_shared<ov::op::v0::Parameter>(
            key_cache_in.get_element_type(), ov::PartialShape{static_cast<int64_t>(num_blocks), 1, static_cast<int64_t>(page_size), static_cast<int64_t>(Di)});
        raw_k_cache->set_friendly_name(paged_attn->get_friendly_name() + "_indexer_raw_k_cache");
        auto summary_cache = std::make_shared<ov::op::v0::Parameter>(
            key_cache_in.get_element_type(),
            ov::PartialShape{static_cast<int64_t>(num_blocks), 1, static_cast<int64_t>(summaries_per_page), static_cast<int64_t>(Di)});
        summary_cache->set_friendly_name(paged_attn->get_friendly_name() + "_indexer_summary_cache");
        auto position_cache = std::make_shared<ov::op::v0::Parameter>(
            ov::element::i32, ov::PartialShape{static_cast<int64_t>(num_blocks), static_cast<int64_t>(summaries_per_page)});
        position_cache->set_friendly_name(paged_attn->get_friendly_name() + "_indexer_position_cache");
        model->add_parameters({raw_k_cache, summary_cache, position_cache});

        ov::OutputVector indexer_args(ov::op::internal::PagedQSAIndexer::InputIdx::COUNT);
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::Q_IDX] = q_idx_flat;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::K_IDX] = k_idx_flat;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::POSITION_IDS] = position_ids;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::ROTARY_COS_SIN] = facts.rope_const;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::SUMMARY_CACHE] = summary_cache;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::INDEXER_RAW_K_CACHE] = raw_k_cache;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::Q_NORM_WEIGHT] = facts.q_norm_const;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::K_NORM_WEIGHT] = facts.k_norm_const;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDICES] = block_indices_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::PAST_LENS] = past_lens_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::INDEXER_POSITION_CACHE] = position_cache;
        auto paged_indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(indexer_args, indexer_config);
        new_ops.push_back(paged_indexer);

        ov::OutputVector attn_args(ov::op::internal::SparsePA::InputIdx::COUNT);
        attn_args[ov::op::internal::SparsePA::InputIdx::QUERY] = query_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::KEY] = key_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::VALUE] = value_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::KEY_CACHE] = key_cache_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::VALUE_CACHE] = value_cache_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::BLOCK_INDICES] = block_indices_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::PAST_LENS] = past_lens_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::SEL_INDICES] = paged_indexer->output(0);
        attn_args[ov::op::internal::SparsePA::InputIdx::SEL_COUNT] = paged_indexer->output(1);
        attn_args[ov::op::internal::SparsePA::InputIdx::QB_START] = subsequence_begins_in;
        auto sparse_pa = std::make_shared<ov::op::internal::SparsePA>(attn_args, attn_config, paged_attn->get_out_type(0));
        new_ops.push_back(sparse_pa);

        ov::copy_runtime_info(ov::NodeVector{root, paged_attn}, new_ops);
        // paged_attn has 3 outputs vs. sparse_pa's 1 (checked above that 1/2 are unused),
        // so only output(0)'s consumers are rewired -- a plain ov::replace_node requires
        // matching output counts and would throw here.
        for (auto& input : paged_attn->output(0).get_target_inputs()) {
            input.replace_source_output(sparse_pa->output(0));
        }
        changed = true;
    }
    return changed;
}

}  // namespace ov::intel_gpu
