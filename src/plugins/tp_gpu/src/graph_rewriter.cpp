// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_rewriter.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tp_gpu/op/tp_all_reduce.hpp"
#include "tp_gpu/op/tp_gather.hpp"
#include "tp_gpu/tp_debug.hpp"
#include "tp_gpu/tp_device_coordinator.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace tp_gpu {

namespace {

/// Ops a projection's result may pass through on its way to attention or to
/// the next projection.
/// The traversal stops at everything else, which is what keeps a search local
/// to one transformer block.
bool is_transparent(const ov::Node* node) {
    return ov::is_type_any_of<
        // layout / shape
        ov::op::v1::Reshape, ov::op::v0::Unsqueeze, ov::op::util::SqueezeBase,
        ov::op::v1::Transpose, ov::op::v0::Convert, ov::op::v0::Concat,
        ov::op::v8::Slice, ov::op::v1::StridedSlice,
        ov::op::v1::Split, ov::op::v1::VariadicSplit,
        ov::op::util::BroadcastBase,
        ov::op::util::GatherBase,
        // elementwise math and every activation derived from it
        ov::op::util::BinaryElementwiseArithmetic,
        ov::op::util::UnaryElementwiseArithmetic,
        // activations that derive straight from Op
        ov::op::v4::Swish, ov::op::v0::PRelu>(node);
}

/// True for a MatMul that is an actual linear layer, i.e. one whose second
/// input is a weight rather than another activation.
///
/// Rotary embeddings compute their frequencies with `inv_freq @ position_ids`,
/// hanging off the very elementwise ops the traversal walks through; requiring
/// a fully constant-derived weight tells the two apart for any weight shape
/// and decompression chain. A projection whose weight is not constant (a live
/// LoRA adapter) cannot be sharded at compile time anyway, and the callers
/// report it as "no projection found" rather than mis-sharding it.
bool is_linear_layer(const ov::Node* node) {
    return ov::is_type<ov::op::v0::MatMul>(node) &&
           ov::op::util::is_on_path<ov::op::v0::Constant>(node->input_value(1));
}

/// Walks backwards from `output` and reports the linear layers it runs into.
std::vector<std::shared_ptr<ov::Node>> nearest_producer_matmuls(const ov::Output<ov::Node>& output) {
    std::vector<std::shared_ptr<ov::Node>> found;
    std::unordered_set<ov::Node*> visited;

    ov::op::util::visit_path(
        output.get_node(),
        visited,
        [](ov::Node*) {},
        [&](ov::Node* node) {
            if (ov::is_type<ov::op::v0::MatMul>(node)) {
                if (is_linear_layer(node))
                    found.push_back(node->shared_from_this());
                return true;
            }
            return !is_transparent(node);
        });
    return found;
}

/// Forward counterpart: the closest linear layer reachable from `start`.
std::shared_ptr<ov::Node> nearest_consumer_matmul(const std::shared_ptr<ov::Node>& start) {
    std::shared_ptr<ov::Node> found;
    std::unordered_set<ov::Node*> visited;

    ov::op::util::visit_path_forward(
        start.get(),
        visited,
        [](ov::Node*) {},
        [&](ov::Node* node) {
            if (node == start.get())
                return false;  // always descend from the anchor itself
            if (found)
                return true;
            if (ov::is_type<ov::op::v0::MatMul>(node)) {
                if (is_linear_layer(node))
                    found = node->shared_from_this();
                return true;
            }
            return !is_transparent(node);
        });
    return found;
}

/// Slice of `dim` owned by `rank`.
///
/// The remainder is spread over the first ranks, one extra unit each, rather
/// than piling up on the last. `granularity` is the indivisible unit of the
/// split -- a quantization group, typically -- which the stored weight cannot
/// express being cut through.
struct Shard {
    int64_t offset;
    int64_t size;
};

Shard shard_of(int64_t dim, uint32_t rank, uint32_t world_size, int64_t granularity = 1) {
    OPENVINO_ASSERT(granularity > 0 && dim % granularity == 0,
                    "[TP_GPU] Cannot split ", dim, " elements in units of ", granularity);

    const auto r = static_cast<int64_t>(rank);
    const int64_t units = dim / granularity;
    const int64_t base = units / world_size;
    const int64_t remainder = units % world_size;
    const int64_t offset = (r * base + std::min(r, remainder)) * granularity;
    const int64_t size = (base + (r < remainder ? 1 : 0)) * granularity;
    return {offset, size};
}

/// Slice of a projection's sharded dimension that belongs to `rank`.
///
/// Attention projections are split by KV head.
/// MLP projections are split directly.
Shard projection_shard(const ShardingPlan& plan,
                       const ShardingPlan::LinearDesc& desc,
                       int64_t full,
                       uint32_t rank,
                       uint32_t world_size,
                       int64_t granularity) {
    const int64_t head_dim = plan.head_dim;
    const int64_t group = plan.num_heads / plan.num_kv_heads;
    const auto kv = shard_of(plan.num_kv_heads, rank, world_size);

    switch (desc.role) {
    case ShardingPlan::LinearDesc::Q_PROJ:
    case ShardingPlan::LinearDesc::O_PROJ:
        // Both sides of attention: `full` spans all query heads.
        OPENVINO_ASSERT(full == static_cast<int64_t>(plan.num_heads) * head_dim,
                        "[TP_GPU] Projection '", desc.matmul_name, "' spans ", full,
                        " features, expected ", plan.num_heads, " query heads x ", head_dim);
        return {kv.offset * group * head_dim, kv.size * group * head_dim};

    case ShardingPlan::LinearDesc::K_PROJ:
    case ShardingPlan::LinearDesc::V_PROJ:
        OPENVINO_ASSERT(full == static_cast<int64_t>(plan.num_kv_heads) * head_dim,
                        "[TP_GPU] Projection '", desc.matmul_name, "' spans ", full,
                        " features, expected ", plan.num_kv_heads, " KV heads x ", head_dim);
        return {kv.offset * head_dim, kv.size * head_dim};

    default:
        return shard_of(full, rank, world_size, granularity);
    }
}

/// A bias sitting on a projection, i.e. `Add(matmul, const_subgraph)`.
struct BiasAdd {
    std::shared_ptr<ov::Node> node;  ///< nullptr when the projection has no bias
    size_t port = 0;                 ///< input index of the bias on `node`
};

/// Finds the bias Add of a projection, if it has one.
///
/// The other operand has to be entirely constant-derived; that is what tells a
/// real bias apart from a residual connection, whose subgraph reaches the model
/// inputs.
BiasAdd find_bias_add(const std::shared_ptr<ov::Node>& matmul) {
    for (const auto& target : ov::op::util::get_node_target_inputs(matmul)) {
        auto add = target.get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v1::Add>(add))
            continue;
        const size_t bias_port = 1 - target.get_index();
        if (!ov::op::util::is_on_path<ov::op::v0::Constant>(add->input_value(bias_port)))
            continue;
        return {add, bias_port};
    }
    return {};
}

/// The output the rest of the graph sees for a projection: the bias Add when
/// there is one, the MatMul itself otherwise.
ov::Output<ov::Node> projection_output(const std::shared_ptr<ov::Node>& matmul) {
    auto bias = find_bias_add(matmul);
    return bias.node ? bias.node->output(0) : matmul->output(0);
}

/// Records a projection in the plan. Returns false when the MatMul was already
/// classified, which callers use to detect overlapping matches.
bool add_linear(ShardingPlan& plan,
                std::unordered_set<const ov::Node*>& classified,
                const std::shared_ptr<ov::Node>& matmul,
                int layer_idx,
                ShardingPlan::LinearDesc::Role role,
                bool is_column_parallel) {
    if (!classified.insert(matmul.get()).second)
        return false;

    ShardingPlan::LinearDesc desc;
    desc.matmul_name = matmul->get_friendly_name();
    desc.layer_idx = layer_idx;
    desc.role = role;
    desc.is_column_parallel = is_column_parallel;
    desc.has_bias = find_bias_add(matmul).node != nullptr;
    plan.linears.push_back(desc);
    return true;
}

/// Reads the head layout off the Reshape that splits a projection output into
/// [.., heads, head_dim]. Returns false when no such Reshape is found.
bool read_head_layout(const std::shared_ptr<ov::Node>& proj, int& heads, int& head_dim) {
    for (const auto& target : projection_output(proj).get_target_inputs()) {
        auto consumer = target.get_node()->shared_from_this();
        if (!ov::is_type<ov::op::v1::Reshape>(consumer))
            continue;
        auto shape_const =
            ov::as_type_ptr<ov::op::v0::Constant>(consumer->input(1).get_source_output().get_node_shared_ptr());
        if (!shape_const)
            continue;
        auto data = shape_const->cast_vector<int64_t>();
        if (data.size() < 4)
            continue;
        heads = static_cast<int>(data[2]);
        head_dim = static_cast<int>(data[3]);
        return true;
    }
    return false;
}

/// Which weight axis carries the output features, and which the input ones.
struct WeightAxes {
    int64_t out_features;
    int64_t in_features;
};

WeightAxes weight_axes(const std::shared_ptr<ov::Node>& matmul) {
    auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(matmul);
    OPENVINO_ASSERT(mm != nullptr, "[TP_GPU] '", matmul->get_friendly_name(), "' is not a MatMul");
    OPENVINO_ASSERT(!mm->get_transpose_a(),
                    "[TP_GPU] Projection '", matmul->get_friendly_name(),
                    "' transposes its activation input, which is not supported");

    const auto& weight_shape = matmul->input(1).get_partial_shape();
    OPENVINO_ASSERT(weight_shape.rank().is_static() && weight_shape.rank().get_length() == 2,
                    "[TP_GPU] Projection '", matmul->get_friendly_name(),
                    "' has a weight of shape ", weight_shape,
                    "; only 2D weights can be sharded (batched MatMul is not supported yet)");

    return mm->get_transpose_b() ? WeightAxes{0, 1} : WeightAxes{1, 0};
}

/// Output feature count of a linear layer.
int weight_out_features(const std::shared_ptr<ov::Node>& matmul) {
    const auto& dim = matmul->input(1).get_partial_shape()[weight_axes(matmul).out_features];
    return dim.is_static() ? static_cast<int>(dim.get_length()) : 0;
}

struct MlpPattern {
    std::shared_ptr<ov::Node> root;   ///< the down projection
    std::shared_ptr<ov::Node> gate;   ///< nullptr for a single-branch MLP
    std::shared_ptr<ov::Node> up;
};

/// down( act(gate(x)) * up(x) ) -- LLaMA/Mistral/Qwen2 style.
///
/// Each projection may carry a bias, so the pattern accepts either the bare
/// MatMul or the MatMul followed by an Add. `wrap_type` matches by
/// `is_castable`, so naming the elementwise base class covers every activation
/// derived from it.
MlpPattern make_gated_mlp_pattern() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;
    auto src = any_input();
    auto gate = wrap_type<ov::op::v0::MatMul>({src, any_input()});
    auto up = wrap_type<ov::op::v0::MatMul>({src, any_input()});
    auto gate_out = gate | wrap_type<ov::op::v1::Add>({gate, any_input()});
    auto up_out = up | wrap_type<ov::op::v1::Add>({up, any_input()});
    auto act = wrap_type<ov::op::util::UnaryElementwiseArithmetic, ov::op::v4::Swish>({gate_out});
    auto mul = wrap_type<ov::op::v1::Multiply>({act, up_out}) | wrap_type<ov::op::v1::Multiply>({up_out, act});
    return {wrap_type<ov::op::v0::MatMul>({mul, any_input()}), gate, up};
}

/// down( act(up(x)) ) -- older single-branch style.
MlpPattern make_simple_mlp_pattern() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;
    auto up = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    auto up_out = up | wrap_type<ov::op::v1::Add>({up, any_input()});
    auto act = wrap_type<ov::op::util::UnaryElementwiseArithmetic, ov::op::v4::Swish>({up_out});
    return {wrap_type<ov::op::v0::MatMul>({act, any_input()}), nullptr, up};
}

}  // namespace

// ---------------------------------------------------------------------------
// analyze()
// ---------------------------------------------------------------------------

ShardingPlan GraphRewriter::analyze(const std::shared_ptr<const ov::Model>& model) {
    ShardingPlan plan;

    // ---- 1) Attention anchors, in topological order -> one per layer ----
    std::vector<std::shared_ptr<ov::Node>> attentions;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::PagedAttentionExtension>(op)) {
            attentions.push_back(op);
        }
    }
    if (!attentions.empty()) {
        plan.attention_backend = ShardingPlan::AttentionBackend::PA;
    } else {
        for (const auto& op : model->get_ordered_ops()) {
            if (ov::is_type<ov::op::v13::ScaledDotProductAttention>(op)) {
                attentions.push_back(op);
            }
        }
        plan.attention_backend = ShardingPlan::AttentionBackend::SDPA;
    }
    OPENVINO_ASSERT(!attentions.empty(),
                    "[TP_GPU] Could not identify transformer layers for TP: the model contains "
                    "neither PagedAttentionExtension nor ScaledDotProductAttention. "
                    "Model may not be supported.");

    plan.num_layers = static_cast<int>(attentions.size());

    std::unordered_set<const ov::Node*> classified;
    std::shared_ptr<ov::Node> q_proj_0, k_proj_0, mlp_col_0;

    // ---- 2) Attention projections ----
    //
    // Inputs 0/1/2 of both SDPA and PagedAttention are query/key/value, and
    // the output feeds the out projection.
    static constexpr ShardingPlan::LinearDesc::Role kQkvRoles[3] = {
        ShardingPlan::LinearDesc::Q_PROJ,
        ShardingPlan::LinearDesc::K_PROJ,
        ShardingPlan::LinearDesc::V_PROJ,
    };
    static constexpr const char* kQkvNames[3] = {"query", "key", "value"};

    for (int layer = 0; layer < plan.num_layers; ++layer) {
        const auto& attention = attentions[layer];

        for (size_t i = 0; i < 3; ++i) {
            OPENVINO_ASSERT(i < attention->get_input_size(),
                            "[TP_GPU] Attention node '", attention->get_friendly_name(),
                            "' has no ", kQkvNames[i], " input");

            auto producers = nearest_producer_matmuls(attention->input(i).get_source_output());
            OPENVINO_ASSERT(producers.size() == 1,
                            "[TP_GPU] Expected exactly one projection feeding the ", kQkvNames[i],
                            " input of '", attention->get_friendly_name(), "', found ",
                            producers.size(),
                            ". A projection is a MatMul with a constant weight; none is found when "
                            "the weight is produced at runtime (a LoRA adapter, for instance). "
                            "Model may not be supported.");

            add_linear(plan, classified, producers.front(), layer, kQkvRoles[i], true);
            if (layer == 0 && i == 0)
                q_proj_0 = producers.front();
            if (layer == 0 && i == 1)
                k_proj_0 = producers.front();
        }

        auto out_proj = nearest_consumer_matmul(attention);
        OPENVINO_ASSERT(out_proj != nullptr,
                        "[TP_GPU] Could not find the out projection following '",
                        attention->get_friendly_name(), "'. Model may not be supported.");
        add_linear(plan, classified, out_proj, layer, ShardingPlan::LinearDesc::O_PROJ, false);
    }

    // ---- 3) MLP blocks ----
    //
    // Matched by shape rather than walked, because the gate/up/down topology is
    // rigid: both projections must grow from the same source, which the shared
    // `src` label in the pattern encodes directly. Walking in topological order
    // lets us attribute each block to the attention that precedes it.
    const auto gated_mlp = make_gated_mlp_pattern();
    const auto simple_mlp = make_simple_mlp_pattern();
    ov::pass::pattern::Matcher gated_matcher(gated_mlp.root, "TPGatedMLP");
    ov::pass::pattern::Matcher simple_matcher(simple_mlp.root, "TPSimpleMLP");

    std::unordered_map<const ov::Node*, int> attention_layer;
    for (int layer = 0; layer < plan.num_layers; ++layer)
        attention_layer[attentions[layer].get()] = layer;

    auto record_mlp = [&](const std::shared_ptr<ov::Node>& down,
                          const std::vector<std::shared_ptr<ov::Node>>& column_projections,
                          int layer) {
        for (const auto& projection : column_projections) {
            add_linear(plan, classified, projection, layer, ShardingPlan::LinearDesc::UP_PROJ, true);
            if (layer == 0 && !mlp_col_0)
                mlp_col_0 = projection;
        }
        add_linear(plan, classified, down, layer, ShardingPlan::LinearDesc::DOWN_PROJ, false);
    };

    int current_layer = -1;
    for (const auto& op : model->get_ordered_ops()) {
        auto anchor = attention_layer.find(op.get());
        if (anchor != attention_layer.end()) {
            current_layer = anchor->second;
            continue;
        }
        if (current_layer < 0 || !is_linear_layer(op.get()) || classified.count(op.get()) != 0)
            continue;

        if (gated_matcher.match(op->output(0))) {
            const auto& matched = gated_matcher.get_pattern_value_map();
            auto gate = matched.at(gated_mlp.gate).get_node_shared_ptr();
            auto up = matched.at(gated_mlp.up).get_node_shared_ptr();
            // The shared label should already guarantee this; check anyway so a
            // pattern change can never silently shard unrelated projections.
            if (gate->input_value(0) == up->input_value(0)) {
                record_mlp(op, {gate, up}, current_layer);
                continue;
            }
        }

        if (simple_matcher.match(op->output(0))) {
            const auto& matched = simple_matcher.get_pattern_value_map();
            record_mlp(op, {matched.at(simple_mlp.up).get_node_shared_ptr()}, current_layer);
        }
    }

    // ---- 4) Model dimensions, read off the layer-0 projections ----
    OPENVINO_ASSERT(q_proj_0 && k_proj_0,
                    "[TP_GPU] Could not locate the layer-0 query/key projections");
    OPENVINO_ASSERT(mlp_col_0,
                    "[TP_GPU] Could not locate the layer-0 MLP projections");

    // The query projection produces num_heads * head_dim features, which is
    // what the post-attention Reshape collapses back to and therefore what has
    // to be localized per rank.
    plan.hidden_size = weight_out_features(q_proj_0);
    plan.intermediate_size = weight_out_features(mlp_col_0);

    OPENVINO_ASSERT(read_head_layout(q_proj_0, plan.num_heads, plan.head_dim),
                    "[TP_GPU] Could not determine the head layout: the query projection of layer 0 "
                    "is not followed by a Reshape with a static shape constant");

    const int kv_features = weight_out_features(k_proj_0);
    if (plan.head_dim > 0)
        plan.num_kv_heads = kv_features / plan.head_dim;

    OPENVINO_ASSERT(plan.num_heads > 0 && plan.head_dim > 0 && plan.num_kv_heads > 0 &&
                        plan.hidden_size > 0 && plan.intermediate_size > 0,
                    "[TP_GPU] Incomplete model geometry: layers=", plan.num_layers,
                    " heads=", plan.num_heads, " kv_heads=", plan.num_kv_heads,
                    " head_dim=", plan.head_dim, " hidden=", plan.hidden_size,
                    " intermediate=", plan.intermediate_size);

    // Sharding splits attention by KV head and lets the query heads follow, so
    // the grouping has to divide evenly and the query projection has to span
    // exactly num_heads * head_dim features.
    OPENVINO_ASSERT(plan.num_heads % plan.num_kv_heads == 0,
                    "[TP_GPU] ", plan.num_heads, " query heads do not group evenly over ",
                    plan.num_kv_heads, " KV heads");
    OPENVINO_ASSERT(plan.hidden_size == plan.num_heads * plan.head_dim,
                    "[TP_GPU] The query projection produces ", plan.hidden_size,
                    " features, which does not match ", plan.num_heads, " heads x ",
                    plan.head_dim, " head_dim");

    // ---- 5) The vocabulary projection ----
    //
    // It is not part of any layer, so the walk above never classified it.
    // Accept it only inits plain form: straight into a Result, nothing else reading it,
    // so that a bias or a second consumer leaves the graph alone.
    for (const auto& result : model->get_results()) {
        auto producer = result->input_value(0).get_node_shared_ptr();
        if (ov::is_type<ov::op::v0::Convert>(producer)) {
            producer = producer->input_value(0).get_node_shared_ptr();
        }
        auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(producer);
        if (!matmul || classified.count(matmul.get()) != 0) {
            continue;
        }
        if (matmul->output(0).get_target_inputs().size() != 1) {
            continue;
        }
        const auto axes = weight_axes(matmul);
        const auto& vocab = matmul->input_value(1).get_partial_shape()[axes.out_features];
        if (!vocab.is_static()) {
            continue;
        }
        plan.lm_head_name = matmul->get_friendly_name();
        plan.lm_head_vocab = vocab.get_length();
        break;
    }

    return plan;
}

// ---------------------------------------------------------------------------
// rewrite() — helpers
// ---------------------------------------------------------------------------

namespace {

/// Pre-slice a Constant's data along `axis` to [start, end).
/// Returns a new smaller Constant.
///
/// When the slice is contiguous in memory (outer == 1, i.e. axis == 0 or all
/// preceding dims are 1), the returned Constant is a zero-copy VIEW into the
/// original Constant's buffer — no allocation, no memcpy.
///
/// When the slice is non-contiguous (outer > 1), the data is copied into a
/// new buffer.
std::shared_ptr<ov::op::v0::Constant> pre_slice_constant(const std::shared_ptr<ov::op::v0::Constant>& c,
                                                         int64_t axis,
                                                         int64_t start,
                                                         int64_t end) {
    auto shape = c->get_shape();
    auto et = c->get_element_type();
    auto ndim = static_cast<int64_t>(shape.size());

    OPENVINO_ASSERT(axis >= 0 && axis < ndim, "[TP] Axis out of range");
    OPENVINO_ASSERT(start >= 0 && end > start &&
                    end <= static_cast<int64_t>(shape[axis]),
                    "[TP] Slice range [", start, ":", end, ") out of bounds for dim ", shape[axis]);

    ov::Shape out_shape = shape;
    out_shape[axis] = static_cast<size_t>(end - start);

    size_t outer = 1;
    for (int64_t d = 0; d < axis; ++d)
        outer *= shape[d];
    size_t inner = 1;
    for (int64_t d = axis + 1; d < ndim; ++d)
        inner *= shape[d];

    size_t bitwidth = et.bitwidth();
    size_t full_row_elems = shape[axis] * inner;
    size_t shard_elems = static_cast<size_t>(end - start) * inner;
    size_t offset_elems = static_cast<size_t>(start) * inner;

    auto elems_to_bytes = [&](size_t elems) -> size_t {
        if (bitwidth >= 8)
            return elems * (bitwidth / 8);
        size_t epb = 8 / bitwidth;
        OPENVINO_ASSERT(elems % epb == 0,
                        "[TP] Element count ", elems, " not byte-aligned for ", et);
        return elems / epb;
    };

    std::shared_ptr<ov::op::v0::Constant> sliced;

    if (outer == 1) {
        // Contiguous slice — zero-copy view into the original Constant's buffer.
        size_t offset_bytes = elems_to_bytes(offset_elems);
        auto data_ptr = static_cast<const char*>(c->get_data_ptr()) + offset_bytes;

        // Keep the original Constant alive via shared_ptr aliasing.
        auto owner = std::shared_ptr<void>(c);

        sliced = std::make_shared<ov::op::v0::Constant>(
            et, out_shape,
            static_cast<const void*>(data_ptr),
            owner);
    } else {
        // Non-contiguous — must gather strided rows into a new buffer.
        size_t full_row_bytes = elems_to_bytes(full_row_elems);
        size_t shard_bytes = elems_to_bytes(shard_elems);
        size_t offset_bytes = elems_to_bytes(offset_elems);

        std::vector<uint8_t> buf(outer * shard_bytes);
        auto src = reinterpret_cast<const uint8_t*>(c->get_data_ptr());
        for (size_t i = 0; i < outer; ++i) {
            std::memcpy(buf.data() + i * shard_bytes,
                        src + i * full_row_bytes + offset_bytes,
                        shard_bytes);
        }
        sliced = std::make_shared<ov::op::v0::Constant>(et, out_shape, buf.data());
    }

    sliced->get_rt_info() = c->get_rt_info();  // preserve decompression markers
    return sliced;
}

/// How an output axis of a Reshape maps back onto its input, ignoring where the
/// slice boundaries happen to fall.
///
/// When the output dimension spans several consecutive input dimensions
/// (e.g. Reshape [N, G, D] -> [N, G*D]) the outermost of them is reported,
/// together with how many output elements one step along it covers. That step
/// -- `trailing` -- is the granularity any slice has to respect.
struct ReshapeAxisSpan {
    int64_t axis = -1;      // axis in the input tensor, or -1 when unmappable
    int64_t full_size = 0;  // size of that axis in the input
    int64_t trailing = 1;   // output elements per unit of that axis
};

ReshapeAxisSpan map_reshape_axis_span(const std::shared_ptr<ov::Node>& reshape,
                                      int64_t output_axis,
                                      int64_t full_size) {
    auto out_ps = reshape->get_output_partial_shape(0);
    auto in_ps = reshape->input(0).get_source_output().get_partial_shape();
    if (!in_ps.rank().is_static() || !out_ps.rank().is_static())
        return {};

    size_t out_before = 1;
    for (int64_t d = 0; d < output_axis && d < out_ps.rank().get_length(); ++d) {
        if (!out_ps[d].is_static())
            return {};
        out_before *= out_ps[d].get_length();
    }

    size_t cum = 1;
    for (int64_t d = 0; d < in_ps.rank().get_length(); ++d) {
        if (!in_ps[d].is_static())
            return {};
        if (cum == out_before) {
            int64_t in_dim = static_cast<int64_t>(in_ps[d].get_length());
            // Case 1: exact 1-to-1 dimension match.
            if (in_dim == full_size)
                return {d, in_dim, 1};
            // Case 2: the output dim spans several input dims, e.g. [G, D] -> [G*D].
            int64_t trailing = 1;
            for (int64_t dd = d + 1; dd < in_ps.rank().get_length(); ++dd) {
                if (!in_ps[dd].is_static())
                    return {};
                trailing *= static_cast<int64_t>(in_ps[dd].get_length());
                if (in_dim * trailing == full_size)
                    return {d, in_dim, trailing};
            }
            return {};
        }
        cum *= in_ps[d].get_length();
    }
    return {};
}

/// Map an axis and a slice backwards through a Reshape.
///
/// Returns {input_axis, input_full_size, input_start, input_end}, or
/// {-1, 0, 0, 0} when the mapping fails or the boundaries do not line up with
/// the inner-dimension granularity.
struct ReshapeAxisMapping {
    int64_t axis;        // axis in the input tensor (or -1)
    int64_t full_size;   // size of that axis in the input
    int64_t start;       // adjusted start index
    int64_t end;         // adjusted end index
};

ReshapeAxisMapping map_reshape_axis(const std::shared_ptr<ov::Node>& reshape,
                                    int64_t output_axis,
                                    int64_t full_size,
                                    int64_t out_start,
                                    int64_t out_end) {
    const auto span = map_reshape_axis_span(reshape, output_axis, full_size);
    if (span.axis < 0)
        return {-1, 0, 0, 0};
    if (out_start % span.trailing != 0 || out_end % span.trailing != 0)
        return {-1, 0, 0, 0};
    return {span.axis, span.full_size, out_start / span.trailing, out_end / span.trailing};
}

/// Coarsest step a slice along `axis` has to respect for the decompression
/// chain to be pre-sliceable, or 0 when the chain cannot be pre-sliced at all.
///
/// Two things constrain it. Quantized weights are stored grouped -- a
/// [N, groups, group_size] constant reshaped to [N, K] -- so a slice may not cut
/// a group in half. And sub-byte constants pack several elements per byte, so a
/// slice may not stop mid-byte either; that bites hardest on the per-group
/// zero-points, whose innermost dimension is 1, meaning an odd number of u4
/// groups is unrepresentable.
///
/// Violating either is not an error, it just forces the fallback: a runtime
/// Slice over the *decompressed* weight, which the GPU plugin then has to
/// constant-fold. On a multi-billion parameter model that costs tens of
/// seconds of compilation, so it is well worth aligning the shards instead.
int64_t chain_granularity(const ov::Output<ov::Node>& output, int64_t axis, int64_t full_size) {
    auto node = output.get_node_shared_ptr();

    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        const auto& shape = constant->get_shape();
        // Mirror pre_slice_chain(): constants that do not carry this axis are
        // left untouched and therefore constrain nothing.
        if (axis < 0 || axis >= static_cast<int64_t>(shape.size()) ||
            static_cast<int64_t>(shape[axis]) != full_size)
            return 1;

        const size_t bitwidth = constant->get_element_type().bitwidth();
        if (bitwidth >= 8)
            return 1;

        int64_t inner = 1;
        for (size_t d = axis + 1; d < shape.size(); ++d)
            inner *= static_cast<int64_t>(shape[d]);

        // `inner` elements come with every step along `axis`; we need whole bytes.
        const int64_t per_byte = static_cast<int64_t>(8 / bitwidth);
        return per_byte / std::gcd(per_byte, inner);
    }

    if (ov::is_type<ov::op::v0::Convert>(node))
        return chain_granularity(node->input(0).get_source_output(), axis, full_size);

    if (ov::is_type_any_of<ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Add>(node)) {
        const int64_t lhs = chain_granularity(node->input(0).get_source_output(), axis, full_size);
        const int64_t rhs = chain_granularity(node->input(1).get_source_output(), axis, full_size);
        if (lhs == 0 || rhs == 0)
            return 0;
        return std::lcm(lhs, rhs);
    }

    if (ov::is_type<ov::op::v1::Reshape>(node)) {
        const auto span = map_reshape_axis_span(node, axis, full_size);
        if (span.axis < 0)
            return 0;
        const int64_t inner =
            chain_granularity(node->input(0).get_source_output(), span.axis, span.full_size);
        if (inner == 0)
            return 0;
        return span.trailing * inner;
    }

    return 0;
}

/// Read-only check: can the decompression chain be pre-sliced?
/// Returns true if every Reshape in the chain can map the axis cleanly.
bool can_pre_slice_chain(ov::Output<ov::Node> output,
                         int64_t axis,
                         int64_t full_size,
                         int64_t start,
                         int64_t end) {
    auto node = output.get_node_shared_ptr();

    if (ov::is_type<ov::op::v0::Constant>(node))
        return true;

    if (ov::is_type<ov::op::v0::Convert>(node)) {
        return can_pre_slice_chain(node->input(0).get_source_output(), axis, full_size, start, end);
    } else if (ov::is_type_any_of<ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Add>(node)) {
        return can_pre_slice_chain(node->input(0).get_source_output(), axis, full_size, start, end) &&
               can_pre_slice_chain(node->input(1).get_source_output(), axis, full_size, start, end);
    } else if (ov::is_type<ov::op::v1::Reshape>(node)) {
        auto m = map_reshape_axis(node, axis, full_size, start, end);
        if (m.axis < 0) return false;
        return can_pre_slice_chain(node->input(0).get_source_output(), m.axis, m.full_size, m.start, m.end);
    }
    return false;  // unknown op
}

/// Pre-slice all Constants in the decompression chain.
void pre_slice_chain(ov::Input<ov::Node> input,
                     int64_t axis,
                     int64_t full_size,
                     int64_t start,
                     int64_t end) {
    auto node = input.get_source_output().get_node_shared_ptr();

    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        auto shape = c->get_shape();
        if (axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
            static_cast<int64_t>(shape[axis]) == full_size) {
            auto sliced = pre_slice_constant(c, axis, start, end);
            sliced->set_friendly_name(c->get_friendly_name() + "_shard");
            input.replace_source_output(sliced->output(0));
        }
        return;
    }

    if (ov::is_type<ov::op::v0::Convert>(node)) {
        pre_slice_chain(node->input(0), axis, full_size, start, end);
    } else if (ov::is_type_any_of<ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Add>(node)) {
        pre_slice_chain(node->input(0), axis, full_size, start, end);
        pre_slice_chain(node->input(1), axis, full_size, start, end);
    } else if (ov::is_type<ov::op::v1::Reshape>(node)) {
        auto m = map_reshape_axis(node, axis, full_size, start, end);
        pre_slice_chain(node->input(0), m.axis, m.full_size, m.start, m.end);

        // Patch the Reshape shape constant.
        // When the output dim maps 1:1 to an input dim, patch at `axis`.
        // When it spans multiple input dims, patch at `m.axis` in the
        // Reshape's shape vector (the outermost input dim that changed).
        auto sc = ov::as_type_ptr<ov::op::v0::Constant>(
            node->input(1).get_source_output().get_node_shared_ptr());
        if (sc) {
            auto data = sc->cast_vector<int64_t>();
            if (static_cast<size_t>(axis) < data.size() && data[axis] == full_size) {
                // 1:1 case: output dim matches exactly
                data[axis] = end - start;
                auto new_sc = ov::op::v0::Constant::create(
                    sc->get_element_type(), sc->get_shape(), data);
                node->input(1).replace_source_output(new_sc->output(0));
            } else if (m.full_size != full_size &&
                       static_cast<size_t>(m.axis) < data.size() &&
                       data[m.axis] == m.full_size) {
                // Multi-dim case: patch the outermost input dim
                data[m.axis] = m.end - m.start;
                auto new_sc = ov::op::v0::Constant::create(
                    sc->get_element_type(), sc->get_shape(), data);
                node->input(1).replace_source_output(new_sc->output(0));
            }
        }
    }
}

/// Fallback: insert a runtime Slice node on the given output.
ov::Output<ov::Node> insert_weight_slice(const ov::Output<ov::Node>& weight_output,
                                         int64_t axis,
                                         int64_t start,
                                         int64_t end) {
    auto begin_c = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{start});
    auto end_c = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{end});
    auto step_c = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{1});
    auto axes_c = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{axis});

    auto slice = std::make_shared<ov::op::v8::Slice>(weight_output, begin_c, end_c, step_c, axes_c);
    return slice->output(0);
}

/// Splits the bias of a column-parallel projection the same way as its weight.
///
/// Row-parallel projections deliberately keep the full bias on every rank: the
/// AllReduce is inserted on the MatMul output, so the Add runs *after* it and
/// has to see the complete bias for the ranks to stay in agreement. Adding a
/// sharded bias there, or adding it on rank 0 only, would make the ranks
/// diverge right after the projection.
void shard_column_parallel_bias(const std::shared_ptr<ov::Node>& matmul,
                                int64_t out_features,
                                const Shard& slice) {
    auto bias = find_bias_add(matmul);
    if (!bias.node)
        return;

    auto bias_output = bias.node->input_value(bias.port);
    const auto& bias_shape = bias_output.get_partial_shape();
    if (!bias_shape.rank().is_static() || bias_shape.rank().get_length() == 0)
        return;

    // The bias broadcasts against [.., out_features], so the feature count is
    // its last dimension. A bias that is 1 there applies to every output alike
    // and stays correct without splitting.
    const int64_t axis = bias_shape.rank().get_length() - 1;
    if (!bias_shape[axis].is_static() || bias_shape[axis].get_length() != out_features)
        return;

    const int64_t start = slice.offset;
    const int64_t end = slice.offset + slice.size;
    if (can_pre_slice_chain(bias_output, axis, out_features, start, end)) {
        pre_slice_chain(bias.node->input(bias.port), axis, out_features, start, end);
    } else {
        bias.node->input(bias.port).replace_source_output(
            insert_weight_slice(bias_output, axis, start, end));
    }
}

/// Replace a Reshape's shape constant, updating one element.
void patch_reshape_constant(const std::shared_ptr<ov::Node>& reshape,
                            size_t shape_idx,
                            int64_t new_value) {
    auto shape_node = reshape->input(1).get_source_output().get_node()->shared_from_this();
    auto shape_const = ov::as_type_ptr<ov::op::v0::Constant>(shape_node);
    if (!shape_const)
        return;

    auto data = shape_const->cast_vector<int64_t>();
    if (shape_idx >= data.size())
        return;

    data[shape_idx] = new_value;
    auto new_const = ov::op::v0::Constant::create(shape_const->get_element_type(),
                                                  shape_const->get_shape(),
                                                  data);
    reshape->input(1).replace_source_output(new_const->output(0));
}

/// Collect the KV cache variables that get sharded along the kv-head axis.
///
/// A variable qualifies when its id names a KV cache and its shape is the
/// canonical [batch, kv_heads, seq, head_dim] with a static kv_heads matching
/// the model-wide count.  Anything else is left alone, which also means it
/// stays replicated across ranks -- the runtime relies on this same predicate
/// to decide how to gather and scatter state, so the two must not drift apart.
std::vector<std::shared_ptr<ov::op::util::Variable>> collect_kv_cache_variables(const ov::Model& model,
                                                                                int num_kv_heads) {
    std::vector<std::shared_ptr<ov::op::util::Variable>> result;
    std::unordered_set<std::string> seen;

    for (const auto& op : model.get_ops()) {
        std::shared_ptr<ov::op::util::Variable> variable;
        if (auto rv = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            variable = rv->get_variable();
        } else if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            variable = assign->get_variable();
        }
        if (!variable)
            continue;

        const auto& info = variable->get_info();
        const auto& var_id = info.variable_id;
        if (var_id.find("past_key_values") == std::string::npos &&
            var_id.find("key") == std::string::npos &&
            var_id.find("value") == std::string::npos)
            continue;

        const auto& shape = info.data_shape;
        if (!shape.rank().is_static() || shape.rank().get_length() != 4)
            continue;
        if (!shape[1].is_static() || shape[1].get_length() != static_cast<int64_t>(num_kv_heads))
            continue;

        // ReadValue and Assign share one Variable object.
        if (seen.insert(var_id).second)
            result.push_back(variable);
    }
    return result;
}

// ---------------------------------------------------------------------------
// rewrite() — steps
// ---------------------------------------------------------------------------

using NameMap = std::unordered_map<std::string, std::shared_ptr<ov::Node>>;

/// What the steps need besides the model.
struct RewriteContext {
    const ShardingPlan& plan;
    const NameMap& name_map;
    uint32_t rank;
    uint32_t tp_degree;
    int64_t local_q_heads;
    int64_t local_kv_heads;
};

void log_step(const char* step, const std::string& outcome) {
    TP_LOG_INFO << "[TP][rewrite]   " << std::left << std::setw(22) << step << outcome << std::endl;
}

/// Coarsest slice step the MLP projections can all respect.
///
/// They share one intermediate dimension -- gate/up produce it, down consumes
/// it -- so all three have to be split at the same offsets. Their storage
/// layouts differ, so the step has to be the coarsest of the three; each
/// projection's own would leave gate/up and down disagreeing.
int64_t mlp_shard_granularity(const ShardingPlan& plan, const NameMap& name_map) {
    int64_t granularity = 1;
    for (const auto& desc : plan.linears) {
        if (desc.role != ShardingPlan::LinearDesc::GATE_PROJ &&
            desc.role != ShardingPlan::LinearDesc::UP_PROJ &&
            desc.role != ShardingPlan::LinearDesc::DOWN_PROJ)
            continue;
        auto it = name_map.find(desc.matmul_name);
        if (it == name_map.end())
            continue;

        const auto axes = weight_axes(it->second);
        const int64_t axis = desc.is_column_parallel ? axes.out_features : axes.in_features;
        const auto& dim = it->second->input(1).get_partial_shape()[axis];
        if (!dim.is_static())
            continue;
        const int64_t step =
            chain_granularity(it->second->input(1).get_source_output(), axis, dim.get_length());
        if (step > 0 && dim.get_length() % step == 0)
            granularity = std::lcm(granularity, step);
    }
    return granularity;
}

/// Step 1: give every projection this rank's band of its weight.
void shard_projection_weights(const RewriteContext& ctx) {
    const int64_t mlp_granularity = mlp_shard_granularity(ctx.plan, ctx.name_map);

    size_t pre_sliced = 0;
    size_t runtime_sliced = 0;
    size_t missing = 0;
    size_t biases = 0;

    for (const auto& desc : ctx.plan.linears) {
        auto it = ctx.name_map.find(desc.matmul_name);
        if (it == ctx.name_map.end()) {
            ++missing;
            continue;
        }
        auto matmul = it->second;

        auto weight_output = matmul->input(1).get_source_output();
        const auto axes = weight_axes(matmul);
        const int64_t axis = desc.is_column_parallel ? axes.out_features : axes.in_features;
        const auto& full_dim = weight_output.get_partial_shape()[axis];
        OPENVINO_ASSERT(full_dim.is_static(),
                        "[TP_GPU] Projection '", desc.matmul_name,
                        "' has a dynamic weight dimension and cannot be sharded");
        const int64_t full = full_dim.get_length();

        // Attention projections are split by head and ignore this; a dimension
        // that does not divide by the shared step falls back below.
        const int64_t granularity = (full % mlp_granularity == 0) ? mlp_granularity : 1;

        const auto slice = projection_shard(ctx.plan, desc, full, ctx.rank, ctx.tp_degree, granularity);
        const int64_t start = slice.offset;
        const int64_t end = slice.offset + slice.size;

        if (can_pre_slice_chain(weight_output, axis, full, start, end)) {
            pre_slice_chain(matmul->input(1), axis, full, start, end);
            ++pre_sliced;
        } else {
            // A runtime Slice leaves the GPU plugin to constant-fold the weight,
            // which dominates compile time.
            ++runtime_sliced;
            matmul->input(1).replace_source_output(
                insert_weight_slice(weight_output, axis, start, end));
            TP_LOG_DEBUG << "[TP][rewrite]     '" << desc.matmul_name
                         << "' falls back to a runtime Slice on axis " << axis << " [" << start
                         << ":" << end << ") of " << full << std::endl;
        }

        if (desc.is_column_parallel) {
            shard_column_parallel_bias(matmul, full, slice);
            if (desc.has_bias)
                ++biases;
        }
    }

    std::ostringstream outcome;
    outcome << "sharded " << (pre_sliced + runtime_sliced) << "/" << ctx.plan.linears.size()
            << " (pre-sliced " << pre_sliced << ", runtime slice " << runtime_sliced
            << ", biases " << biases << ")";
    if (missing != 0)
        outcome << " -- " << missing << " NOT FOUND in the cloned graph";
    log_step("projection weights", outcome.str());

    if (runtime_sliced != 0) {
        // Not gated by the verbosity: this is a compile-time cliff, and whoever
        // hits it needs to know without having been told to look.
        TP_WARN_ALWAYS << "[TP_GPU] Warning: rank " << ctx.rank << ": " << runtime_sliced << " of "
                       << ctx.plan.linears.size()
                       << " weights could not be pre-sliced and fall back to a runtime Slice"
                       << " (this dominates compile time)";
    }
}

/// Step 2: localize the head count in the Reshape after each q/k/v projection,
/// which turns [B, S, features] into [B, S, heads, head_dim].
void localize_qkv_head_counts(const RewriteContext& ctx) {
    size_t patched = 0;
    size_t without_reshape = 0;

    for (const auto& desc : ctx.plan.linears) {
        if (desc.role != ShardingPlan::LinearDesc::Q_PROJ &&
            desc.role != ShardingPlan::LinearDesc::K_PROJ &&
            desc.role != ShardingPlan::LinearDesc::V_PROJ)
            continue;

        auto it = ctx.name_map.find(desc.matmul_name);
        if (it == ctx.name_map.end())
            continue;

        const int64_t new_heads = (desc.role == ShardingPlan::LinearDesc::Q_PROJ)
                                      ? ctx.local_q_heads
                                      : ctx.local_kv_heads;

        bool found = false;
        for (const auto& target : projection_output(it->second).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v1::Reshape>(consumer)) {
                patch_reshape_constant(consumer, /*shape_idx=*/2, new_heads);
                found = true;
                ++patched;
            }
        }
        if (!found)
            ++without_reshape;
    }

    std::ostringstream outcome;
    outcome << "patched " << patched << " reshapes (q=" << ctx.local_q_heads
            << " kv=" << ctx.local_kv_heads << ")";
    if (without_reshape != 0)
        outcome << " -- " << without_reshape << " projections had none";
    log_step("qkv head counts", outcome.str());
}

/// Step 3 (SDPA only): localize the broadcast that expands KV heads up to the
/// query head count.
///
/// It appears both as a multiply by a ones tensor and as a bidirectional
/// Broadcast; the shared matcher covers both, rooted at the Reshape that
/// merges the expanded heads back into [B, num_heads, S, head_dim].
void localize_gqa_broadcast(const ov::Model& model, const RewriteContext& ctx) {
    if (ctx.plan.attention_backend != ShardingPlan::AttentionBackend::SDPA) {
        log_step("gqa broadcast", "skipped -- attention is PagedAttention");
        return;
    }
    if (ctx.plan.num_kv_heads == ctx.plan.num_heads) {
        log_step("gqa broadcast", "not needed -- no grouped-query attention");
        return;
    }

    auto kv_bcst = ov::op::util::match_multi_query_bcst(ov::pass::pattern::any_input());
    ov::pass::pattern::Matcher matcher(std::get<0>(kv_bcst), "TPMultiQueryBcst");

    size_t patched = 0;
    for (const auto& op : model.get_ops()) {
        if (!ov::is_type<ov::op::v1::Reshape>(op) || !matcher.match(op->output(0)))
            continue;
        auto shape_const =
            ov::as_type_ptr<ov::op::v0::Constant>(op->input(1).get_source_output().get_node_shared_ptr());
        if (!shape_const)
            continue;
        auto shape_data = shape_const->cast_vector<int64_t>();
        if (shape_data.size() == 4 && shape_data[1] == static_cast<int64_t>(ctx.plan.num_heads)) {
            patch_reshape_constant(op, /*shape_idx=*/1, ctx.local_q_heads);
            ++patched;
        }
    }

    OPENVINO_ASSERT(patched > 0,
                    "[TP_GPU] The model uses grouped-query attention (", ctx.plan.num_kv_heads,
                    " KV heads for ", ctx.plan.num_heads,
                    " Q heads) but no KV broadcast was found to re-shape");

    log_step("gqa broadcast",
             "patched " + std::to_string(patched) + " reshapes to " +
                 std::to_string(ctx.local_q_heads) + " heads");
}

/// Step 3 (PagedAttention only): localize the kv head count kept in rt_info.
///
/// The reshapes around PagedAttention are relative, so sharding the
/// projections already localizes them. The kv head count is not in the graph
/// at all: the conversion records it in rt_info, and both the pass that sizes
/// the cache and the GPU plugin read it from there while deriving the query
/// head count from the already-sharded operand.
void localize_paged_attention_heads(const ov::Model& model, const RewriteContext& ctx) {
    if (ctx.plan.attention_backend != ShardingPlan::AttentionBackend::PA) {
        log_step("paged attention", "skipped -- attention is SDPA");
        return;
    }

    static constexpr const char* kv_head_keys[] = {"num_k_heads", "num_v_heads"};

    size_t localized = 0;
    for (const auto& op : model.get_ops()) {
        if (!ov::is_type<ov::op::PagedAttentionExtension>(op))
            continue;
        auto& rt_info = op->get_rt_info();
        for (const auto* key : kv_head_keys) {
            auto entry = rt_info.find(key);
            if (entry == rt_info.end())
                continue;
            OPENVINO_ASSERT(entry->second.as<int64_t>() == static_cast<int64_t>(ctx.plan.num_kv_heads),
                            "[TP_GPU] '", op->get_friendly_name(), "' declares ",
                            entry->second.as<int64_t>(), " for '", key, "' where the model has ",
                            ctx.plan.num_kv_heads, " kv heads");
            entry->second = static_cast<size_t>(ctx.local_kv_heads);
            ++localized;
        }
    }

    // Absent on models whose conversion did not record the geometry; the
    // plugin then reads it off the cache tensor, which is already local.
    const size_t expected = std::size(kv_head_keys) * static_cast<size_t>(ctx.plan.num_layers);
    OPENVINO_ASSERT(localized == 0 || localized == expected,
                    "[TP_GPU] Localized ", localized, " kv head counts over ", ctx.plan.num_layers,
                    " PagedAttention layers; expected ", expected, ". Model may not be supported.");

    std::ostringstream outcome;
    if (localized == 0)
        outcome << "no rt_info head counts -- the plugin reads them off the cache";
    else
        outcome << "localized " << localized << " rt_info entries to " << ctx.local_kv_heads
                << " kv heads";
    log_step("paged attention", outcome.str());
}

/// Step 4: localize the Reshape that flattens [B, S, heads, head_dim] back
/// into hidden_size before the out projection.
///
/// Exports that build this shape with ShapeOf+Concat are already relative and
/// need no patching; only a baked constant does.
void localize_attention_output_reshape(const RewriteContext& ctx) {
    const int64_t hidden = static_cast<int64_t>(ctx.plan.hidden_size);
    const int64_t local_hidden = ctx.local_q_heads * static_cast<int64_t>(ctx.plan.head_dim);

    size_t patched = 0;
    size_t relative = 0;

    for (const auto& desc : ctx.plan.linears) {
        if (desc.role != ShardingPlan::LinearDesc::O_PROJ)
            continue;

        auto it = ctx.name_map.find(desc.matmul_name);
        if (it == ctx.name_map.end())
            continue;

        // Scoped to the out projection's own producer chain, so an unrelated
        // Reshape carrying the same value is never touched.
        std::shared_ptr<ov::Node> src = it->second->input(0).get_source_output().get_node_shared_ptr();
        for (int hops = 0; hops < 4 && src && !ov::is_type<ov::op::v1::Reshape>(src); ++hops) {
            if (ov::is_type<ov::op::v0::Convert>(src)) {
                src = src->input(0).get_source_output().get_node_shared_ptr();
            } else {
                src.reset();
                break;
            }
        }
        if (!src || !ov::is_type<ov::op::v1::Reshape>(src))
            continue;

        auto shape_const = ov::as_type_ptr<ov::op::v0::Constant>(
            src->input(1).get_source_output().get_node_shared_ptr());
        if (!shape_const) {
            ++relative;
            continue;
        }

        auto data = shape_const->cast_vector<int64_t>();
        bool changed = false;
        for (auto& v : data) {
            if (v == hidden) {
                v = local_hidden;
                changed = true;
            }
        }
        if (!changed)
            continue;

        src->input(1).replace_source_output(
            ov::op::v0::Constant::create(shape_const->get_element_type(), shape_const->get_shape(), data)
                ->output(0));
        ++patched;
    }

    std::ostringstream outcome;
    outcome << "patched " << patched << " reshapes " << hidden << " -> " << local_hidden;
    if (relative != 0)
        outcome << ", " << relative << " already relative";
    log_step("attn out reshape", outcome.str());
}

/// Step 5: shrink the kv-head axis of every KV cache variable.
void localize_kv_cache_variables(const ov::Model& model, const RewriteContext& ctx) {
    size_t updated = 0;
    for (const auto& variable : collect_kv_cache_variables(model, ctx.plan.num_kv_heads)) {
        auto info = variable->get_info();
        info.data_shape[1] = ctx.local_kv_heads;
        variable->update(info);
        ++updated;
    }

    std::ostringstream outcome;
    outcome << "resized " << updated << " variables to " << ctx.local_kv_heads << " kv heads";
    log_step("kv cache variables", outcome.str());
}

/// Step 6: localize the kv-head constant in each KV cache init subgraph.
///
/// Every ReadValue has an init chain with a constant of its own, even when
/// they share a binary offset in the IR, so stopping at the first leaves the
/// rest holding the full head count.
void localize_kv_cache_init(const ov::Model& model, const RewriteContext& ctx) {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> kv_init_consts;

    for (const auto& op : model.get_ops()) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!c || c->get_shape() != ov::Shape{1})
            continue;
        auto data = c->cast_vector<int64_t>();
        if (data.size() != 1 || data[0] != static_cast<int64_t>(ctx.plan.num_kv_heads))
            continue;

        // Confirm the chain Constant -> Concat -> Broadcast -> ReadValue, so a
        // constant that merely carries the same number is left alone.
        bool feeds_kv_init = false;
        for (const auto& target : c->output(0).get_target_inputs()) {
            auto concat = target.get_node()->shared_from_this();
            if (!ov::is_type<ov::op::v0::Concat>(concat))
                continue;
            for (const auto& ct : concat->output(0).get_target_inputs()) {
                auto broadcast = ct.get_node()->shared_from_this();
                if (!ov::is_type_any_of<ov::op::v1::Broadcast, ov::op::v3::Broadcast>(broadcast))
                    continue;
                for (const auto& bt : broadcast->output(0).get_target_inputs()) {
                    if (ov::is_type<ov::op::v6::ReadValue>(bt.get_node())) {
                        feeds_kv_init = true;
                        break;
                    }
                }
                if (feeds_kv_init) break;
            }
            if (feeds_kv_init) break;
        }
        if (feeds_kv_init)
            kv_init_consts.push_back(c);
    }

    for (auto& c : kv_init_consts) {
        c->output(0).replace(ov::op::v0::Constant::create(c->get_element_type(), c->get_shape(),
                                                          std::vector<int64_t>{ctx.local_kv_heads})
                                 ->output(0));
    }

    std::ostringstream outcome;
    outcome << "patched " << kv_init_consts.size() << " init constants";
    log_step("kv cache init", outcome.str());
}

uint32_t row_parallel_count(const ShardingPlan& plan) {
    return static_cast<uint32_t>(
        std::count_if(plan.linears.begin(), plan.linears.end(), [](const auto& desc) {
            return !desc.is_column_parallel;
        }));
}

/// Step 7: put a TPAllReduce on every row-parallel projection, whose output is
/// a partial sum across ranks.
void insert_all_reduce(const RewriteContext& ctx) {
    uint32_t collective_id = 0;
    size_t missing = 0;

    for (const auto& desc : ctx.plan.linears) {
        if (desc.is_column_parallel)
            continue;

        auto it = ctx.name_map.find(desc.matmul_name);
        if (it == ctx.name_map.end()) {
            ++missing;
            continue;
        }
        auto matmul = it->second;

        auto ar = std::make_shared<ov::tp_gpu::op::TPAllReduce>(matmul->output(0),
                                                                collective_id++,
                                                                ctx.rank,
                                                                ctx.tp_degree);
        ar->set_friendly_name("tp_allreduce/" + desc.matmul_name);

        auto targets = matmul->output(0).get_target_inputs();
        for (const auto& target : targets) {
            if (target.get_node() == ar.get())
                continue;
            target.replace_source_output(ar->output(0));
        }
    }

    std::ostringstream outcome;
    outcome << "inserted " << collective_id << " collectives";
    if (missing != 0)
        outcome << " -- " << missing << " row-parallel projections NOT FOUND";
    log_step("all-reduce", outcome.str());
}

/// Step 8: split the vocabulary projection and gather the slices into rank 0.
void shard_vocabulary_projection(const RewriteContext& ctx, const TPConfig& config) {
    if (!GraphRewriter::shards_lm_head(ctx.plan, static_cast<int>(ctx.tp_degree), config)) {
        std::ostringstream why;
        if (ctx.plan.lm_head_name.empty())
            why << "skipped -- no vocabulary projection found";
        else if (config.disable_lm_head_sharding())
            why << "skipped -- disabled by configuration";
        else
            why << "skipped -- vocabulary " << ctx.plan.lm_head_vocab << " does not divide by "
                << ctx.tp_degree;
        log_step("vocabulary split", why.str());
        return;
    }

    auto it = ctx.name_map.find(ctx.plan.lm_head_name);
    OPENVINO_ASSERT(it != ctx.name_map.end(),
                    "[TP_GPU] The vocabulary projection '", ctx.plan.lm_head_name,
                    "' disappeared from the cloned model");
    auto matmul = it->second;

    const auto axes = weight_axes(matmul);
    const int64_t vocab = ctx.plan.lm_head_vocab;
    const int64_t band = vocab / ctx.tp_degree;
    const int64_t start = band * ctx.rank;
    const int64_t end = start + band;

    auto weight_output = matmul->input(1).get_source_output();
    const bool pre_sliced = can_pre_slice_chain(weight_output, axes.out_features, vocab, start, end);
    if (pre_sliced) {
        pre_slice_chain(matmul->input(1), axes.out_features, vocab, start, end);
    } else {
        matmul->input(1).replace_source_output(
            insert_weight_slice(weight_output, axes.out_features, start, end));
    }
    matmul->validate_and_infer_types();

    const uint32_t gather_id = row_parallel_count(ctx.plan);
    auto gather = std::make_shared<ov::tp_gpu::op::TPGather>(matmul->output(0),
                                                             gather_id,
                                                             ctx.rank,
                                                             ctx.tp_degree,
                                                             /*axis=*/-1);
    gather->set_friendly_name(matmul->get_friendly_name() + "/tp_gather");

    auto targets = matmul->output(0).get_target_inputs();
    for (const auto& target : targets) {
        if (target.get_node() == gather.get())
            continue;
        target.replace_source_output(gather->output(0));
    }

    std::ostringstream outcome;
    outcome << "split " << vocab << " -> " << band << " rows ("
            << (pre_sliced ? "pre-sliced" : "runtime slice") << "), gathered by collective "
            << gather_id;
    log_step("vocabulary split", outcome.str());
}

}  // namespace

// ---------------------------------------------------------------------------
// rewrite()
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> GraphRewriter::rewrite(const std::shared_ptr<const ov::Model>& model,
                                                  const ShardingPlan& plan,
                                                  uint32_t rank,
                                                  uint32_t tp_degree,
                                                  const TPConfig& config) {
    OPENVINO_ASSERT(tp_degree >= 2, "[TP_GPU] tp_degree must be >= 2");

    // KV heads are the unit attention is split by, so they -- not the query
    // heads -- cap how many ranks the model can use.
    OPENVINO_ASSERT(static_cast<int>(tp_degree) <= plan.num_kv_heads,
                    "[TP_GPU] The model has ", plan.num_kv_heads, " KV heads, so it can be split "
                    "across at most that many ranks; TP_SIZE=", tp_degree,
                    " was requested. Replicating KV heads across ranks is not supported yet.");

    auto cloned = model->clone();

    // Query heads follow their KV head, so that each rank owns whole
    // grouped-query groups and every query head keeps the key head it attends
    // through. Deriving the two independently would break that pairing
    // whenever num_kv_heads is not a multiple of tp_degree.
    const auto kv_shard = shard_of(plan.num_kv_heads, rank, tp_degree);

    NameMap name_map;
    for (const auto& op : cloned->get_ops()) {
        name_map[op->get_friendly_name()] = op;
    }

    const RewriteContext ctx{plan,
                             name_map,
                             rank,
                             tp_degree,
                             kv_shard.size * (plan.num_heads / plan.num_kv_heads),
                             kv_shard.size};

    TP_LOG_INFO << "[TP][rewrite] rank " << rank << " of " << tp_degree << ": " << plan.num_layers
                << " layers, " << plan.linears.size() << " projections, " << plan.num_heads << "/"
                << plan.num_kv_heads << " q/kv heads -> " << ctx.local_q_heads << "/"
                << ctx.local_kv_heads << std::endl;

    // The steps are independent; the order only keeps the shapes consistent
    // for the validation below.
    shard_projection_weights(ctx);
    localize_qkv_head_counts(ctx);
    localize_gqa_broadcast(*cloned, ctx);
    localize_paged_attention_heads(*cloned, ctx);
    localize_attention_output_reshape(ctx);
    localize_kv_cache_variables(*cloned, ctx);
    localize_kv_cache_init(*cloned, ctx);
    insert_all_reduce(ctx);
    shard_vocabulary_projection(ctx, config);

    cloned->validate_nodes_and_infer_types();

    return cloned;
}

bool GraphRewriter::shards_lm_head(const ShardingPlan& plan, int tp_degree, const TPConfig& config) {
    if (plan.lm_head_name.empty() || tp_degree <= 1 || config.disable_lm_head_sharding()) {
        return false;
    }
    return plan.lm_head_vocab % static_cast<int64_t>(tp_degree) == 0;
}

int GraphRewriter::count_collectives(const ShardingPlan& plan, int tp_degree, const TPConfig& config) {
    return static_cast<int>(row_parallel_count(plan)) + (shards_lm_head(plan, tp_degree, config) ? 1 : 0);
}

std::vector<std::string> GraphRewriter::sharded_state_ids(const std::shared_ptr<const ov::Model>& model,
                                                          const ShardingPlan& plan) {
    std::vector<std::string> ids;
    for (const auto& variable : collect_kv_cache_variables(*model, plan.num_kv_heads)) {
        ids.push_back(variable->get_info().variable_id);
    }
    return ids;
}

}  // namespace tp_gpu
}  // namespace ov
