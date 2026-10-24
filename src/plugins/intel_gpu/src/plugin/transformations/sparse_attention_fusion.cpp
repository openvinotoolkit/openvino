// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_fusion.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/sparse_pa.hpp"
#include "openvino/op/sparse_sdpa.hpp"
#include "ov_ops/rms.hpp"

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

/// The QSA indexer score pattern: scores = ReduceSum(ReLU(MatMul(index_query, k_block,
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

bool is_ancestor_of(const ov::Output<ov::Node>& start, const ov::Node* target, size_t max_depth) {
    return find_ancestor_if(start, max_depth, nullptr, [target](ov::Node* n) {
        return n == target;
    }) != nullptr;
}

bool is_layout_only(const ov::Node* n) {
    return ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n) || ov::is_type<ov::op::v0::Unsqueeze>(n) ||
           ov::is_type<ov::op::v0::Squeeze>(n) || ov::is_type<ov::op::v1::Transpose>(n);
}

/// `1 + w` gains that were not constant-folded are folded here, so the op receives the
/// gain exactly as the kernel applies it.
std::shared_ptr<ov::op::v0::Constant> fold_one_plus_gain(const std::shared_ptr<ov::op::v0::Constant>& gain) {
    for (const auto& consumer : gain->output(0).get_target_inputs()) {
        auto add = ov::as_type<ov::op::v1::Add>(consumer.get_node());
        if (!add) {
            continue;
        }
        float one = 0.0f;
        if (get_scalar_f32(add->input_value(1 - consumer.get_index()), one) && close(one, 1.0f)) {
            auto values = gain->cast_vector<float>();
            for (auto& v : values) {
                v += 1.0f;
            }
            return ov::op::v0::Constant::create(gain->get_element_type(), gain->get_shape(), values);
        }
    }
    return gain;
}

struct NormFacts {
    std::shared_ptr<ov::op::v0::Constant> gain;
    float eps = 0.0f;
};

/// The GPU lowering of the QSA ops accepts f16 data only.
std::shared_ptr<ov::op::v0::Constant> to_f16_const(const std::shared_ptr<ov::op::v0::Constant>& c) {
    if (c->get_element_type() == ov::element::f16) {
        return c;
    }
    return ov::op::v0::Constant::create(ov::element::f16, c->get_shape(), c->cast_vector<float>());
}

/// Proves one indexer leg's RMSNorm: either an already-fused `ov::op::internal::RMS`
/// (RMSFusion runs earlier in the GPU pipeline) or the decomposed
/// `x * rsqrt(ReduceMean(x^2) + eps) * gain` form.
bool prove_norm(const ov::Output<ov::Node>& leg, ov::Node* stop_at, size_t Di, NormFacts& out) {
    if (auto rms = find_ancestor<ov::op::internal::RMS>(leg, 25, stop_at)) {
        out.eps = static_cast<float>(rms->get_epsilon());
        if (rms->get_input_size() > 1) {
            auto gain = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(rms->input_value(1)).get_node_shared_ptr());
            if (!gain || static_element_count(gain->get_output_partial_shape(0)) != Di) {
                return false;
            }
            out.gain = fold_one_plus_gain(gain);
        } else {
            out.gain = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{Di}, std::vector<float>(Di, 1.0f));
        }
        return true;
    }

    auto gain = ov::as_type_ptr<ov::op::v0::Constant>(find_ancestor_if(leg, 25, stop_at, [Di](ov::Node* n) {
        auto c = ov::as_type<ov::op::v0::Constant>(n);
        return c && static_element_count(c->get_output_partial_shape(0)) == Di;
    }));
    if (!gain) {
        return false;
    }
    auto is_mean = [](ov::Node* n) {
        if (ov::is_type<ov::op::v0::Convert>(n)) {
            n = n->get_input_node_ptr(0);
        }
        return ov::is_type<ov::op::v1::ReduceMean>(n);
    };
    float eps = 0.0f;
    auto eps_add = find_ancestor_if(leg, 25, stop_at, [&](ov::Node* n) {
        if (!ov::is_type<ov::op::v1::Add>(n)) {
            return false;
        }
        for (size_t i = 0; i < 2; ++i) {
            float v = 0.0f;
            if (is_mean(n->get_input_node_ptr(1 - i)) && get_scalar_f32(n->input_value(i), v) && v > 0.0f && v <= 1e-2f) {
                eps = v;
                return true;
            }
        }
        return false;
    });
    if (!eps_add) {
        return false;
    }
    out.eps = eps;
    out.gain = fold_one_plus_gain(gain);
    return true;
}

struct RopeFacts {
    std::shared_ptr<ov::op::v0::Constant> cos_table;  ///< [max_positions, rd], rotate-half row (both halves equal)
    std::shared_ptr<ov::op::v0::Constant> sin_table;  ///< [max_positions, rd], rotate-half row (both halves equal)
    ov::Output<ov::Node> positions;                   ///< the per-token indices the graph gathers rows with
    size_t rotary_dim = 0;
};

/// Proves `rows = Gather(table, positions, axis=0)` on the indexer query leg and the row
/// layout from how `rows` is split on its last axis: step-2 slices mean an interleaved
/// `(cos_j, sin_j)` layout; two `rd/2` halves mean `[cos | sin]`. Either way the table is
/// split into separate full-width (Hugging Face `cat(freqs, freqs)`) cos/sin tables. Any
/// other use bails out.
bool prove_rope(const ov::Output<ov::Node>& idx_q, ov::Node* stop_at, size_t Di, RopeFacts& out) {
    auto gather_node = find_ancestor_if(idx_q, 25, stop_at, [](ov::Node* n) {
        if (!ov::is_type<ov::op::util::GatherBase>(n)) {
            return false;
        }
        auto table = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(n->input_value(0)).get_node_shared_ptr());
        std::vector<int64_t> axis;
        return table && table->get_output_partial_shape(0).size() == 2 && get_i64_vector(n->input_value(2), axis) && axis.size() == 1 &&
               axis[0] == 0;
    });
    if (!gather_node) {
        return false;
    }
    auto table = ov::as_type_ptr<ov::op::v0::Constant>(resolve_constant_source(gather_node->input_value(0)).get_node_shared_ptr());
    const auto table_shape = table->get_shape();
    const size_t rd = table_shape[1];
    if (rd == 0 || rd % 2 != 0 || rd > Di) {
        return false;
    }
    const auto half = static_cast<int64_t>(rd / 2);

    auto last_axis = [](const ov::Node* n, int64_t axis) {
        const auto& rank = n->get_input_partial_shape(0).rank();
        return rank.is_static() && (axis == -1 || axis == rank.get_length() - 1);
    };
    bool halves = false;
    bool interleaved = false;
    std::unordered_set<ov::Node*> visited;
    std::deque<ov::Node*> queue{gather_node.get()};
    while (!queue.empty()) {
        auto* node = queue.front();
        queue.pop_front();
        if (!visited.insert(node).second) {
            continue;
        }
        for (const auto& consumer : node->output(0).get_target_inputs()) {
            auto* n = consumer.get_node();
            if (is_layout_only(n)) {
                queue.push_back(n);
            } else if (auto slice = ov::as_type<ov::op::v8::Slice>(n)) {
                std::vector<int64_t> start, stop, step, axes;
                if (!get_i64_vector(slice->input_value(1), start) || !get_i64_vector(slice->input_value(2), stop) ||
                    !get_i64_vector(slice->input_value(3), step) || start.empty()) {
                    return false;
                }
                if (slice->get_input_size() > 4) {
                    if (!get_i64_vector(slice->input_value(4), axes) || axes.size() != 1 || !last_axis(slice, axes[0])) {
                        return false;
                    }
                } else if (!last_axis(slice, static_cast<int64_t>(start.size()) - 1)) {
                    return false;
                }
                const auto b = start.back(), e = stop.back(), s = step.back();
                if (s == 2 && (b == 0 || b == 1)) {
                    interleaved = true;
                } else if (s == 1 && ((b == 0 && e == half) || (b == half && e >= static_cast<int64_t>(rd)))) {
                    halves = true;
                } else {
                    return false;
                }
            } else if (auto split = ov::as_type<ov::op::v1::Split>(n)) {
                std::vector<int64_t> axis;
                if (split->get_num_splits() != 2 || !get_i64_vector(split->input_value(1), axis) || axis.size() != 1 || !last_axis(split, axis[0])) {
                    return false;
                }
                halves = true;
            } else if (auto vsplit = ov::as_type<ov::op::v1::VariadicSplit>(n)) {
                std::vector<int64_t> axis, lengths;
                if (!get_i64_vector(vsplit->input_value(1), axis) || axis.size() != 1 || !last_axis(vsplit, axis[0]) ||
                    !get_i64_vector(vsplit->input_value(2), lengths) || lengths != std::vector<int64_t>{half, half}) {
                    return false;
                }
                halves = true;
            } else {
                return false;
            }
        }
    }
    if (halves == interleaved) {
        return false;
    }

    const auto values = table->cast_vector<float>();
    const size_t half_rd = rd / 2;
    std::vector<float> cos_values(table_shape[0] * rd);
    std::vector<float> sin_values(table_shape[0] * rd);
    for (size_t p = 0; p < table_shape[0]; ++p) {
        for (size_t j = 0; j < rd; ++j) {
            const size_t f = j % half_rd;
            cos_values[p * rd + j] = halves ? values[p * rd + f] : values[p * rd + 2 * f];
            sin_values[p * rd + j] = halves ? values[p * rd + half_rd + f] : values[p * rd + 2 * f + 1];
        }
    }
    const ov::Shape full_shape{table_shape[0], rd};
    out.cos_table = ov::op::v0::Constant::create(table->get_element_type(), full_shape, cos_values);
    out.sin_table = ov::op::v0::Constant::create(table->get_element_type(), full_shape, sin_values);
    out.positions = gather_node->input_value(1);
    out.rotary_dim = rd;
    return true;
}

/// Per-token `(rope_cos, rope_sin)` rows for `positions` (any shape), each `[..., rd]`.
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> gather_rope_rows(const RopeFacts& rope,
                                                                      const ov::Output<ov::Node>& positions,
                                                                      ov::NodeVector& new_ops) {
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto cos_rows = std::make_shared<ov::op::v8::Gather>(rope.cos_table, positions, axis);
    auto sin_rows = std::make_shared<ov::op::v8::Gather>(rope.sin_table, positions, axis);
    new_ops.insert(new_ops.end(), {axis, cos_rows, sin_rows});
    return {cos_rows->output(0), sin_rows->output(0)};
}

/// Facts shared by both the paged and dense rewrites once the indexer score signature
/// is proven: the shared QK projection, per-leg RMSNorm gains/eps, and the RoPE table +
/// the position indices the source graph gathers it with.
struct SharedIndexerFacts {
    std::shared_ptr<ov::op::v0::MatMul> qk_proj_mm;
    std::shared_ptr<ov::op::v0::Constant> q_norm_const;
    std::shared_ptr<ov::op::v0::Constant> k_norm_const;
    RopeFacts rope;
    float eps = 0.0f;
    size_t Hidx = 0;
    size_t D = 0;
};

/// Proves the shared `[Hidx*Di + Di, D]` INDEXER_QK_WEIGHT projection convention, both
/// legs' RMSNorm (same eps), and the RoPE table/positions. `idx_q`/`idx_k` are the
/// (post-norm/RoPE) operands of the indexer score MatMul.
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

    NormFacts q_norm, k_norm;
    if (!prove_norm(idx_q, idx_q_proj_mm.get(), Di, q_norm) || !prove_norm(idx_k, idx_q_proj_mm.get(), Di, k_norm) || !close(q_norm.eps, k_norm.eps)) {
        return false;
    }
    if (!prove_rope(idx_q, idx_q_proj_mm.get(), Di, out.rope)) {
        return false;
    }

    out.qk_proj_mm = idx_q_proj_mm;
    out.q_norm_const = to_f16_const(q_norm.gain);
    out.k_norm_const = to_f16_const(k_norm.gain);
    out.rope.cos_table = to_f16_const(out.rope.cos_table);
    out.rope.sin_table = to_f16_const(out.rope.sin_table);
    out.eps = q_norm.eps;
    out.Hidx = Hidx;
    out.D = D;
    return true;
}

/// Pre-norm, pre-RoPE indexer Q/K channels sliced off the shared projection's LAST axis
/// only, so every leading (batch/token) axis keeps its runtime extent. QSAIndexer/
/// PagedQSAIndexer apply RMSNorm + RoPE internally; feeding the matched post-norm values
/// would double-apply both.
void split_qk_projection(const ov::Output<ov::Node>& proj,
                          size_t Hidx,
                          size_t Di,
                          ov::NodeVector& new_ops,
                          ov::Output<ov::Node>& q_raw,
                          ov::Output<ov::Node>& k_raw) {
    auto i64c = [&](std::vector<int64_t> v) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
        new_ops.push_back(c);
        return c;
    };
    const auto hidx_di = static_cast<int64_t>(Hidx * Di);
    auto q_slice = std::make_shared<ov::op::v8::Slice>(proj, i64c({0}), i64c({hidx_di}), i64c({1}), i64c({-1}));
    new_ops.push_back(q_slice);
    auto k_slice = std::make_shared<ov::op::v8::Slice>(proj, i64c({hidx_di}), i64c({hidx_di + static_cast<int64_t>(Di)}), i64c({1}), i64c({-1}));
    new_ops.push_back(k_slice);
    q_raw = q_slice->output(0);
    k_raw = k_slice->output(0);
}

std::shared_ptr<ov::op::v0::Parameter> make_in_place_cache(const ov::element::Type& et, const ov::PartialShape& shape, const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(et, shape);
    param->set_friendly_name(name);
    param->output(0).get_tensor().set_names({name});
    param->output(0).get_rt_info()[in_place_cache_rt_key] = true;
    return param;
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
        const auto proj_out = facts.qk_proj_mm->output(0);
        if (proj_out.get_partial_shape().rank().is_dynamic() || proj_out.get_partial_shape().size() != 3 || proj_out.get_element_type() != ov::element::f16) {
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
        if (query_in.get_element_type() != ov::element::f16 || key_in.get_element_type() != ov::element::f16 ||
            value_in.get_element_type() != ov::element::f16) {
            continue;
        }
        if (q_ps.rank().is_dynamic() || q_ps.size() != 4 || k_ps.rank().is_dynamic() || k_ps.size() != 4 || v_ps.rank().is_dynamic() ||
            v_ps.size() != 4 || q_ps[1].is_dynamic() || q_ps[3].is_dynamic() || k_ps[1].is_dynamic() || v_ps[3].is_dynamic()) {
            continue;
        }
        const size_t Hq = static_cast<size_t>(q_ps[1].get_length());
        const size_t Dh = static_cast<size_t>(q_ps[3].get_length());
        const size_t Hkv = static_cast<size_t>(k_ps[1].get_length());
        const size_t Dv = static_cast<size_t>(v_ps[3].get_length());

        // SparseSDPA reproduces causal selected-block attention with the default scale
        // only. A mask is accepted only when derived from this indexer's TopK (i.e. it is
        // the selection mask itself); any other mask, a non-causal call without one, a
        // non-default scale or a sink input is not representable.
        if (sdpa->get_input_size() > 5) {
            continue;
        }
        if (sdpa->get_input_size() > 3) {
            if (!is_ancestor_of(sdpa->input_value(3), root.get(), 100)) {
                continue;
            }
        } else if (!sdpa->get_causal()) {
            continue;
        }
        if (sdpa->get_input_size() > 4) {
            float sdpa_scale = 0.0f;
            if (!get_scalar_f32(sdpa->input_value(4), sdpa_scale) || !close(sdpa_scale, 1.0f / std::sqrt(static_cast<float>(Dh)))) {
                continue;
            }
        }

        // past_len comes from the main K history (its Assign survives this pass), never
        // from the raw-K Variable whose Assign is retired below and would freeze at 0.
        if (key_read_value->get_output_partial_shape(0).rank() != 4) {
            continue;
        }
        auto kv_concat = find_ancestor_if(key_in, 40, nullptr, [&](ov::Node* n) {
            auto concat = ov::as_type<ov::op::v0::Concat>(n);
            return concat && concat->get_output_partial_shape(0).rank() == 4 && (concat->get_axis() == 2 || concat->get_axis() == -2) &&
                   is_ancestor_of(concat->output(0), key_read_value.get(), 4);
        });
        if (!kv_concat) {
            continue;
        }

        // key_block_size `r`: the indexer's own from-scratch block-mean pooling.
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
        if (r == 0) {
            continue;
        }

        ov::op::internal::QSAIndexer::Config indexer_config{};
        indexer_config.key_block_size = r;
        indexer_config.block_topk = static_cast<size_t>(sig.block_topk);
        indexer_config.num_selection_heads = 1;
        indexer_config.indexer_rotary_dim = facts.rope.rotary_dim;
        indexer_config.indexer_head_dim = Di;
        indexer_config.epsilon = facts.eps;
        indexer_config.scale = 1.0f / std::sqrt(static_cast<float>(Di));
        std::string reason;
        // Di % 16: the GPU CM indexer kernels' own constraint (not part of the op contract).
        if (Di % 16 != 0 || !ov::op::internal::QSAIndexer::validate_config(indexer_config, reason)) {
            continue;
        }
        ov::op::internal::SparseSDPA::Config attn_config{};
        attn_config.num_heads = Hq;
        attn_config.num_kv_heads = Hkv;
        attn_config.num_selection_heads = 1;
        attn_config.k_head_size = Dh;
        attn_config.v_head_size = Dv;
        attn_config.key_block_size = r;
        attn_config.query_block_size = 1;
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

        ov::Output<ov::Node> q_raw, k_raw;
        split_qk_projection(proj_out, facts.Hidx, Di, new_ops, q_raw, k_raw);
        auto index_query_in = std::make_shared<ov::op::v1::Reshape>(q_raw, i64c({0, 0, static_cast<int64_t>(facts.Hidx), static_cast<int64_t>(Di)}), true);
        new_ops.push_back(index_query_in);
        auto raw_index_keys_in = std::make_shared<ov::op::v1::Reshape>(k_raw, i64c({0, 0, 1, static_cast<int64_t>(Di)}), true);
        new_ops.push_back(raw_index_keys_in);

        auto proj_shape = std::make_shared<ov::op::v3::ShapeOf>(proj_out, ov::element::i64);
        new_ops.push_back(proj_shape);
        auto batch_dim = std::make_shared<ov::op::v8::Gather>(proj_shape, i64c({0}), i64c({0}));
        new_ops.push_back(batch_dim);
        auto batch_len = std::make_shared<ov::op::v8::Gather>(proj_shape, i64c({0, 1}), i64c({0}));
        new_ops.push_back(batch_len);

        // KEY_LENGTH [B] i32: the main K length after this call (no padding in this graph),
        // shared by QSAIndexer and SparseSDPA.
        auto kv_shape = std::make_shared<ov::op::v3::ShapeOf>(key_in, ov::element::i64);
        new_ops.push_back(kv_shape);
        auto total_len = std::make_shared<ov::op::v8::Gather>(kv_shape, i64c({2}), i64c({0}));
        new_ops.push_back(total_len);
        auto key_length_b = std::make_shared<ov::op::v3::Broadcast>(total_len, batch_dim);
        new_ops.push_back(key_length_b);
        auto key_length = std::make_shared<ov::op::v0::Convert>(key_length_b, ov::element::i32);
        new_ops.push_back(key_length);

        // ROPE_COS/ROPE_SIN [B, L, rd]: the rows the source graph gathers for each token,
        // so non-sequential (mRoPE) positions are preserved.
        auto positions_bl = std::make_shared<ov::op::v1::Reshape>(facts.rope.positions, batch_len, false);
        new_ops.push_back(positions_bl);
        const auto [rope_cos, rope_sin] = gather_rope_rows(facts.rope, positions_bl, new_ops);

        // Indexer state through ReadValue/Assign (functional state interface). Each Variable
        // starts as zeros of its batch-dependent shape; block_index_keys_cache grows on the sequence
        // axis to floor(key_length / r) entries before the call.
        const auto& state_prefix = raw_k_read_value->get_variable_id();
        const auto di = static_cast<int64_t>(Di);
        const auto rd = static_cast<int64_t>(facts.rope.rotary_dim);
        auto f16_zero = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
        new_ops.push_back(f16_zero);
        auto zeros = [&](const ov::OutputVector& dims) {
            auto shape = std::make_shared<ov::op::v0::Concat>(dims, 0);
            auto filled = std::make_shared<ov::op::v3::Broadcast>(f16_zero, shape);
            new_ops.insert(new_ops.end(), {shape, filled});
            return filled->output(0);
        };
        ov::SinkVector state_sinks;
        ov::op::util::VariableVector state_vars;
        auto read_state = [&](const ov::PartialShape& shape, const ov::Output<ov::Node>& init, const std::string& suffix) {
            auto var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{shape, ov::element::f16, state_prefix + suffix});
            auto read = std::make_shared<ov::op::v6::ReadValue>(init, var);
            new_ops.push_back(read);
            state_vars.push_back(var);
            return read;
        };
        auto summary_read = read_state(ov::PartialShape{-1, 1, -1, di}, zeros({batch_dim, i64c({1}), i64c({0}), i64c({di})}), "_qsa_block_index_keys_cache");
        auto raw_read = read_state(ov::PartialShape{-1, 1, static_cast<int64_t>(r) - 1, di},
                                   zeros({batch_dim, i64c({1}), i64c({static_cast<int64_t>(r) - 1}), i64c({di})}),
                                   "_qsa_past_incomplete_raw_k");
        auto rope_read = read_state(ov::PartialShape{-1, 2 * rd}, zeros({batch_dim, i64c({2 * rd})}), "_qsa_past_incomplete_rope");

        auto summary_shape = std::make_shared<ov::op::v3::ShapeOf>(summary_read, ov::element::i64);
        auto summary_entries = std::make_shared<ov::op::v8::Gather>(summary_shape, i64c({2}), i64c({0}));
        auto needed_entries = std::make_shared<ov::op::v1::Divide>(total_len, i64c({static_cast<int64_t>(r)}));
        auto missing = std::make_shared<ov::op::v1::Subtract>(needed_entries, summary_entries);
        auto growth = std::make_shared<ov::op::v1::Maximum>(missing, i64c({0}));
        new_ops.insert(new_ops.end(), {summary_shape, summary_entries, needed_entries, missing, growth});
        auto summary_grown = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{summary_read, zeros({batch_dim, i64c({1}), growth, i64c({di})})}, 2);
        new_ops.push_back(summary_grown);

        using IndexerIn = ov::op::internal::QSAIndexer::InputIdx;
        using IndexerOut = ov::op::internal::QSAIndexer::OutputIdx;
        ov::OutputVector indexer_args(IndexerIn::COUNT);
        indexer_args[IndexerIn::INDEX_QUERY] = index_query_in;
        indexer_args[IndexerIn::RAW_INDEX_KEYS] = raw_index_keys_in;
        indexer_args[IndexerIn::ROPE_COS] = rope_cos;
        indexer_args[IndexerIn::ROPE_SIN] = rope_sin;
        indexer_args[IndexerIn::KEY_LENGTH] = key_length;
        indexer_args[IndexerIn::Q_NORM_WEIGHT] = facts.q_norm_const;
        indexer_args[IndexerIn::K_NORM_WEIGHT] = facts.k_norm_const;
        indexer_args[IndexerIn::BLOCK_INDEX_KEYS_CACHE] = summary_grown;
        indexer_args[IndexerIn::PAST_INCOMPLETE_RAW_K] = raw_read;
        indexer_args[IndexerIn::PAST_INCOMPLETE_ROPE] = rope_read;
        auto qsa_indexer = std::make_shared<ov::op::internal::QSAIndexer>(indexer_args, indexer_config);
        new_ops.push_back(qsa_indexer);
        for (const auto& [out, var] : {std::pair{IndexerOut::BLOCK_INDEX_KEYS_CACHE, state_vars[0]},
                                       std::pair{IndexerOut::PAST_INCOMPLETE_RAW_K, state_vars[1]},
                                       std::pair{IndexerOut::PAST_INCOMPLETE_ROPE, state_vars[2]}}) {
            auto assign = std::make_shared<ov::op::v6::Assign>(qsa_indexer->output(out), var);
            new_ops.push_back(assign);
            state_sinks.push_back(assign);
        }

        ov::OutputVector attn_args(ov::op::internal::SparseSDPA::InputIdx::COUNT_MIN);
        attn_args[ov::op::internal::SparseSDPA::InputIdx::QUERY] = query_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::KEY] = key_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::VALUE] = value_in;
        attn_args[ov::op::internal::SparseSDPA::InputIdx::SELECTED_BLOCK_INDICES] = qsa_indexer->output(IndexerOut::SELECTED_BLOCK_INDICES);
        attn_args[ov::op::internal::SparseSDPA::InputIdx::SELECTED_BLOCK_COUNT] = qsa_indexer->output(IndexerOut::SELECTED_BLOCK_COUNT);
        attn_args[ov::op::internal::SparseSDPA::InputIdx::KEY_LENGTH] = key_length;
        auto sparse_sdpa = std::make_shared<ov::op::internal::SparseSDPA>(attn_args, attn_config, sdpa->get_output_element_type(0));
        new_ops.push_back(sparse_sdpa);

        ov::copy_runtime_info(ov::NodeVector{root, sdpa}, new_ops);
        ov::replace_node(sdpa, sparse_sdpa);
        model->add_variables(state_vars);
        model->add_sinks(state_sinks);

        // Retire any Assign feeding the source raw-K Variable: its history is superseded by
        // the QSAIndexer state above and the decomposed pooling that read it is now dead.
        // The Variable declaration itself is left in place rather than removed -- harmless
        // if unused, and safer than asserting nothing else in the graph still touches it.
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

        // Optional PagedAttentionExtension features SparsePA does not reproduce must be
        // provably disabled (statically empty / zero).
        auto input_empty = [&](size_t idx) {
            if (idx >= paged_attn->get_input_size()) {
                return true;
            }
            const auto& ps = paged_attn->get_input_partial_shape(idx);
            return ps.is_static() && ov::shape_size(ps.to_shape()) == 0;
        };
        auto input_zero = [&](size_t idx) {
            if (idx >= paged_attn->get_input_size()) {
                return true;
            }
            std::vector<int64_t> v;
            return get_i64_vector(paged_attn->input_value(idx), v) && v.size() == 1 && v[0] == 0;
        };
        constexpr size_t sliding_window = 10, alibi_slopes = 11, rotated_block_indices = 14, xattention_threshold = 17, sinks = 20,
                         adaptive_rkv_evictable_sizes = 22, token_type_ids = 25, qq_bias = 26;
        if (!input_zero(sliding_window) || !input_empty(alibi_slopes) || !input_empty(rotated_block_indices) || !input_empty(xattention_threshold) ||
            !input_empty(sinks) || !input_empty(adaptive_rkv_evictable_sizes) || !input_empty(token_type_ids) || !input_empty(qq_bias)) {
            continue;
        }
        // SparsePA reads/writes f16 pages only (no compressed KV cache, no f32 inference).
        if (key_cache_in.get_element_type() != ov::element::f16 || value_cache_in.get_element_type() != ov::element::f16 ||
            query_in.get_element_type() != ov::element::f16 || facts.qk_proj_mm->get_output_element_type(0) != ov::element::f16) {
            continue;
        }

        const auto& key_cache_ps = key_cache_in.get_partial_shape();
        if (key_cache_ps.rank().is_dynamic() || key_cache_ps.size() != 4 || key_cache_ps[1].is_dynamic() || key_cache_ps[2].is_dynamic() ||
            key_cache_ps[3].is_dynamic()) {
            continue;
        }
        const auto num_blocks = key_cache_ps[0];
        const size_t Hkv = static_cast<size_t>(key_cache_ps[1].get_length());
        const size_t page_size = static_cast<size_t>(key_cache_ps[2].get_length());
        const size_t Dh = static_cast<size_t>(key_cache_ps[3].get_length());
        const auto& value_cache_ps = value_cache_in.get_partial_shape();
        if (value_cache_ps.rank().is_dynamic() || value_cache_ps.size() != 4 || value_cache_ps[3].is_dynamic()) {
            continue;
        }
        const size_t Dv = static_cast<size_t>(value_cache_ps[3].get_length());
        const auto& query_ps = query_in.get_partial_shape();
        if (query_ps.rank().is_dynamic() || query_ps.size() != 2 || query_ps[1].is_dynamic()) {
            continue;
        }
        const size_t Hq_Dh = static_cast<size_t>(query_ps[1].get_length());
        if (Dh == 0 || Hq_Dh % Dh != 0) {
            continue;
        }
        const size_t Hq = Hq_Dh / Dh;

        // key_block_size `r`: the indexer's own from-scratch block-mean pooling (same
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
        indexer_config.key_block_size = r;
        indexer_config.block_topk = static_cast<size_t>(sig.block_topk);
        indexer_config.num_selection_heads = 1;
        indexer_config.indexer_rotary_dim = facts.rope.rotary_dim;
        indexer_config.indexer_head_dim = Di;
        indexer_config.epsilon = facts.eps;
        indexer_config.scale = 1.0f / std::sqrt(static_cast<float>(Di));
        std::string reason;
        // Di % 16: the GPU CM indexer kernels' own constraint (not part of the op contract).
        if (Di % 16 != 0 || !ov::op::internal::PagedQSAIndexer::validate_config(indexer_config, reason)) {
            continue;
        }
        ov::op::internal::SparsePA::Config attn_config{};
        attn_config.num_heads = Hq;
        attn_config.num_kv_heads = Hkv;
        attn_config.num_selection_heads = 1;
        attn_config.k_head_size = Dh;
        attn_config.v_head_size = Dv;
        attn_config.key_block_size = r;
        attn_config.query_block_size = 1;
        attn_config.causal = true;
        float attn_scale = 0.0f;
        if (!get_scalar_f32(scale_in, attn_scale)) {
            continue;
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

        ov::Output<ov::Node> q_raw, k_raw;
        split_qk_projection(facts.qk_proj_mm->output(0), facts.Hidx, Di, new_ops, q_raw, k_raw);
        auto flatten_tokens = [&](const ov::Output<ov::Node>& v, size_t heads) {
            auto reshaped = std::make_shared<ov::op::v1::Reshape>(v, i64c({-1, static_cast<int64_t>(heads * Di)}), false);
            new_ops.push_back(reshaped);
            return reshaped->output(0);
        };
        auto index_query_flat = flatten_tokens(q_raw, facts.Hidx);
        auto raw_index_keys_flat = flatten_tokens(k_raw, 1);

        // ROPE_COS/ROPE_SIN [tokens, rd]: the rows the source graph gathers for each
        // token, so non-sequential (mRoPE) positions are preserved.
        auto positions_flat = std::make_shared<ov::op::v1::Reshape>(facts.rope.positions, i64c({-1}), false);
        new_ops.push_back(positions_flat);
        const auto [rope_cos, rope_sin] = gather_rope_rows(facts.rope, positions_flat, new_ops);

        const auto summaries_per_page = static_cast<int64_t>(page_size / r);
        const auto di = static_cast<int64_t>(Di);
        const auto rd = static_cast<int64_t>(facts.rope.rotary_dim);
        const auto& cache_prefix = paged_attn->get_friendly_name();
        auto raw_k_cache = make_in_place_cache(ov::element::f16,
                                               ov::PartialShape{num_blocks, 1, static_cast<int64_t>(page_size), di},
                                               cache_prefix + "_index_key_cache");
        auto block_index_keys_cache =
            make_in_place_cache(ov::element::f16, ov::PartialShape{num_blocks, 1, summaries_per_page, di}, cache_prefix + "_indexer_block_index_keys_cache");
        auto rope_cache =
            make_in_place_cache(ov::element::f16, ov::PartialShape{num_blocks, summaries_per_page, 2 * rd}, cache_prefix + "_index_block_rope_cache");
        model->add_parameters({raw_k_cache, block_index_keys_cache, rope_cache});

        ov::OutputVector indexer_args(ov::op::internal::PagedQSAIndexer::InputIdx::COUNT);
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::INDEX_QUERY] = index_query_flat;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::RAW_INDEX_KEYS] = raw_index_keys_flat;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::ROPE_COS] = rope_cos;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::ROPE_SIN] = rope_sin;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDEX_KEYS_CACHE] = block_index_keys_cache;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::INDEX_KEY_CACHE] = raw_k_cache;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::Q_NORM_WEIGHT] = facts.q_norm_const;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::K_NORM_WEIGHT] = facts.k_norm_const;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDICES] = block_indices_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::PAST_LENS] = past_lens_in;
        indexer_args[ov::op::internal::PagedQSAIndexer::InputIdx::INDEX_BLOCK_ROPE_CACHE] = rope_cache;
        auto paged_indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(indexer_args, indexer_config);
        new_ops.push_back(paged_indexer);

        ov::OutputVector attn_args(ov::op::internal::SparsePA::InputIdx::COUNT_MIN);
        attn_args[ov::op::internal::SparsePA::InputIdx::QUERY] = query_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::KEY] = key_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::VALUE] = value_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::KEY_CACHE] = key_cache_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::VALUE_CACHE] = value_cache_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::BLOCK_INDICES] = block_indices_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::BLOCK_INDICES_BEGINS] = block_indices_begins_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::SUBSEQUENCE_BEGINS] = subsequence_begins_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::PAST_LENS] = past_lens_in;
        attn_args[ov::op::internal::SparsePA::InputIdx::SELECTED_BLOCK_INDICES] = paged_indexer->output(0);
        attn_args[ov::op::internal::SparsePA::InputIdx::SELECTED_BLOCK_COUNT] = paged_indexer->output(1);
        const auto scale_et = scale_in.get_element_type().is_real() ? scale_in.get_element_type() : ov::element::f32;
        auto scale_const = ov::op::v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{attn_scale});
        auto sinks_const = ov::op::v0::Constant::create(query_in.get_element_type(), ov::Shape{0}, std::vector<float>{});  // empty = no sinks
        new_ops.push_back(scale_const);
        new_ops.push_back(sinks_const);
        attn_args[ov::op::internal::SparsePA::InputIdx::SCALE] = scale_const;
        attn_args[ov::op::internal::SparsePA::InputIdx::SINKS] = sinks_const;
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
