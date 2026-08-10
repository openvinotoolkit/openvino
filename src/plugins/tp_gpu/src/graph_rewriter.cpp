// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_rewriter.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "tp_gpu/op/tp_all_reduce.hpp"
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
/// the next projection: layout changes, elementwise math and activations.
///
/// Traversal stops at everything else, which is what keeps a search local to
/// one transformer block.  Normalizations stay opaque because they reduce
/// (`ReduceMean` derives from `ArithmeticReductionKeepDims`), and so do
/// attention and `ReadValue`.
///
/// Base classes are used wherever one exists, so the list does not have to be
/// revisited every time an opset adds a version of the same operation.
bool is_transparent(const ov::Node* node) {
    return ov::is_type_any_of<
        // layout / shape
        ov::op::v1::Reshape, ov::op::v1::Transpose, ov::op::v0::Convert, ov::op::v0::Concat,
        ov::op::v8::Slice, ov::op::v1::StridedSlice, ov::op::v0::Unsqueeze, ov::op::v1::Split,
        ov::op::v1::VariadicSplit,
        ov::op::util::BroadcastBase,  // Broadcast v1, v3
        ov::op::util::SqueezeBase,    // Squeeze v0, v15
        ov::op::util::GatherBase,     // Gather v1, v7, v8
        // elementwise math and every activation derived from it
        ov::op::util::BinaryElementwiseArithmetic,
        ov::op::util::UnaryElementwiseArithmetic,
        // activations that derive straight from Op
        ov::op::v4::Swish, ov::op::v0::PRelu>(node);
}

/// True for a MatMul that is an actual linear layer, i.e. one whose second
/// input is a weight rather than another activation.
///
/// Not every MatMul in a transformer is a projection: rotary embeddings compute
/// their position frequencies with `inv_freq @ position_ids`, which hangs off
/// the very same elementwise ops the traversal walks through.  Its second
/// operand descends from the `position_ids` parameter, so requiring a fully
/// constant-derived weight is enough to tell the two apart -- and unlike a rank
/// check it stays correct for weights of any shape and decompression chain.
///
/// A projection whose weight is not constant (a live LoRA adapter, say) cannot
/// be sharded at compile time anyway; the callers report it as "no projection
/// found" rather than silently mis-sharding it.
bool is_linear_layer(const ov::Node* node) {
    return ov::is_type<ov::op::v0::MatMul>(node) &&
           ov::op::util::is_on_path<ov::op::v0::Constant>(node->input_value(1));
}

/// Walks backwards from `output` and reports the linear layers it runs into.
///
/// The walk stops at every MatMul, so only the nearest producers are returned --
/// that is what keeps the result layer-local.  Recording happens in the stop
/// predicate because `visit_path` does not invoke `func` for nodes it stops at.
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
/// The remainder is spread over the first ranks, one extra unit each, so the
/// load stays as even as possible instead of piling up on the last rank.
///
/// `granularity` is the indivisible unit the split works in -- a quantization
/// group, typically.  Splitting below it would cut a group in half, which the
/// stored weight cannot express.
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
/// Attention projections are split by KV head, never by raw feature count.
/// Under grouped-query attention one KV head serves `num_heads / num_kv_heads`
/// query heads, and a rank has to own whole groups: splitting query and key
/// heads independently would hand a rank query heads whose key heads live
/// somewhere else, which is wrong without ever failing a shape check.
///
/// MLP projections carry no such pairing and are split directly, in whole
/// quantization groups.
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

/// Records a projection in the plan.  Returns false when the MatMul was already
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
/// [.., heads, head_dim].  Returns false when no such Reshape is found.
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
///
/// PyTorch exports linear layers with `transpose_b`, giving a [N, K] weight,
/// but that is an attribute of the op rather than a guarantee: with
/// `transpose_b` off the weight is [K, N] and the two axes swap.  Reading it
/// wrong would shard along the opposite dimension and quietly corrupt results,
/// so the layout is derived here once and everything else goes through it.
///
/// Anything this function cannot describe -- batched (rank > 2) weights as used
/// by MoE experts, or a transposed activation input -- raises instead of being
/// skipped, because a projection that is found but left unsharded desynchronizes
/// the shapes the later steps patch.
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

/// Pattern for an MLP block, kept together so the callback can read the
/// individual projections back out of the match map.
struct MlpPattern {
    std::shared_ptr<ov::Node> root;   ///< the down projection
    std::shared_ptr<ov::Node> gate;   ///< nullptr for a single-branch MLP
    std::shared_ptr<ov::Node> up;
};

/// down( act(gate(x)) * up(x) ) -- LLaMA/Mistral/Qwen2 style.
///
/// Each projection may carry a bias, so the pattern accepts either the bare
/// MatMul or the MatMul followed by an Add.  `wrap_type` matches by
/// `is_castable`, so naming the elementwise base class covers every activation
/// derived from it.
MlpPattern make_gated_mlp_pattern() {
    using namespace ov::pass;           // operator| for pattern alternatives
    using namespace ov::pass::pattern;  // any_input / wrap_type
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
    using namespace ov::pass;           // operator| for pattern alternatives
    using namespace ov::pass::pattern;  // any_input / wrap_type
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
    //
    // Matching is structural on purpose: friendly names survive neither model
    // re-export nor most graph optimizations, so anchoring on them silently
    // shards nothing on anything but a stock HuggingFace export.
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
    // `src` label in the pattern encodes directly.  Walking in topological order
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

    return plan;
}

// ---------------------------------------------------------------------------
// rewrite()  — helpers
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
std::shared_ptr<ov::op::v0::Constant> pre_slice_constant(
    const std::shared_ptr<ov::op::v0::Constant>& c,
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
/// together with how many output elements one step along it covers.  That step
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
/// Two things constrain it.  Quantized weights are stored grouped -- a
/// [N, groups, group_size] constant reshaped to [N, K] -- so a slice may not cut
/// a group in half.  And sub-byte constants pack several elements per byte, so a
/// slice may not stop mid-byte either; that bites hardest on the per-group
/// zero-points, whose innermost dimension is 1, meaning an odd number of u4
/// groups is unrepresentable.
///
/// Violating either is not an error, it just forces the fallback: a runtime
/// Slice over the *decompressed* weight, which the GPU plugin then has to
/// constant-fold.  On a multi-billion parameter model that costs tens of
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
/// PRECONDITION: can_pre_slice_chain() returned true.
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
/// has to see the complete bias for the ranks to stay in agreement.  Adding a
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
    // its last dimension.  A bias that is 1 there applies to every output alike
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

    for (const auto& op : model.get_ordered_ops()) {
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

}  // namespace

// ---------------------------------------------------------------------------
// rewrite()
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> GraphRewriter::rewrite(const std::shared_ptr<const ov::Model>& model,
                                                  const ShardingPlan& plan,
                                                  uint32_t rank,
                                                  uint32_t tp_degree) {
    OPENVINO_ASSERT(tp_degree >= 2, "[TP_GPU] tp_degree must be >= 2");

    // KV heads are the unit attention is split by, so they -- not the query
    // heads -- cap how many ranks the model can use.
    OPENVINO_ASSERT(static_cast<int>(tp_degree) <= plan.num_kv_heads,
                    "[TP_GPU] The model has ", plan.num_kv_heads, " KV heads, so it can be split "
                    "across at most that many ranks; TP_SIZE=", tp_degree,
                    " was requested. Replicating KV heads across ranks is not supported yet.");

    auto cloned = model->clone();

    // Query heads follow their KV head: each rank owns whole grouped-query
    // groups, which keeps every query head paired with the key head it attends
    // through.  Deriving the two independently would silently break that
    // pairing whenever num_kv_heads is not a multiple of tp_degree.
    const auto kv_shard = shard_of(plan.num_kv_heads, rank, tp_degree);
    const int64_t local_kv_heads = kv_shard.size;
    const int64_t local_q_heads = kv_shard.size * (plan.num_heads / plan.num_kv_heads);

    // Build name → node map for the cloned graph.
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> name_map;
    for (const auto& op : cloned->get_ordered_ops()) {
        name_map[op->get_friendly_name()] = op;
    }

    // ------------------------------------------------------------------
    // 1) Shard weights — pre-slice Constants in each decompression chain.
    //
    //    Instead of inserting a runtime Slice op, we trace the weight
    //    decompression chain (Constant → Convert → Subtract → Multiply →
    //    Reshape → Convert) and replace each Constant with a pre-sliced
    //    version containing only this rank's shard.
    //
    //    The last rank extends to the end of the full dimension to handle
    //    cases where the size is not evenly divisible by tp_degree.
    // ------------------------------------------------------------------
    size_t runtime_sliced = 0;

    // Which weight axis a projection is split along, and how long it is.
    auto sharded_axis = [](const std::shared_ptr<ov::Node>& matmul, bool is_column_parallel) {
        const auto axes = weight_axes(matmul);
        const int64_t axis = is_column_parallel ? axes.out_features : axes.in_features;
        return std::make_pair(axis, matmul->input(1).get_partial_shape()[axis]);
    };

    // The MLP projections share one intermediate dimension -- gate/up produce
    // it, down consumes it -- so all three have to be split at exactly the same
    // offsets.  Their storage layouts differ (down is split along the axis whose
    // per-group zero-points sit one element apart, the others are not), so the
    // step has to be the coarsest of them; using each projection's own step
    // would leave gate/up and down disagreeing on the dimension they share.
    int64_t mlp_granularity = 1;
    for (const auto& desc : plan.linears) {
        if (desc.role != ShardingPlan::LinearDesc::GATE_PROJ &&
            desc.role != ShardingPlan::LinearDesc::UP_PROJ &&
            desc.role != ShardingPlan::LinearDesc::DOWN_PROJ)
            continue;
        auto it = name_map.find(desc.matmul_name);
        if (it == name_map.end())
            continue;

        const auto [axis, dim] = sharded_axis(it->second, desc.is_column_parallel);
        if (!dim.is_static())
            continue;
        const int64_t step =
            chain_granularity(it->second->input(1).get_source_output(), axis, dim.get_length());
        if (step > 0 && dim.get_length() % step == 0)
            mlp_granularity = std::lcm(mlp_granularity, step);
    }

    for (const auto& desc : plan.linears) {
        auto it = name_map.find(desc.matmul_name);
        if (it == name_map.end())
            continue;
        auto matmul = it->second;

        auto weight_output = matmul->input(1).get_source_output();
        const auto axes = weight_axes(matmul);

        // Column-parallel splits the output features, row-parallel the input ones.
        const int64_t axis = desc.is_column_parallel ? axes.out_features : axes.in_features;
        const auto& full_dim = weight_output.get_partial_shape()[axis];
        OPENVINO_ASSERT(full_dim.is_static(),
                        "[TP_GPU] Projection '", desc.matmul_name,
                        "' has a dynamic weight dimension and cannot be sharded");
        const int64_t full = full_dim.get_length();

        // Attention projections are split by head and ignore this; MLP ones use
        // the shared step computed above.  When the dimension does not divide by
        // it we cannot align, and the fallback (counted below) takes over.
        const int64_t granularity = (full % mlp_granularity == 0) ? mlp_granularity : 1;

        const auto slice = projection_shard(plan, desc, full, rank, tp_degree, granularity);
        const int64_t start_idx = slice.offset;
        const int64_t end_idx = slice.offset + slice.size;

        if (can_pre_slice_chain(weight_output, axis, full, start_idx, end_idx)) {
            pre_slice_chain(matmul->input(1), axis, full, start_idx, end_idx);
        } else {
            // Falling back here means the weight gets sliced at runtime and the
            // GPU plugin has to constant-fold it, which dominates compile time.
            ++runtime_sliced;
            auto sliced = insert_weight_slice(weight_output, axis, start_idx, end_idx);
            matmul->input(1).replace_source_output(sliced);
        }

        // Column-parallel splits the output features, so a bias on this
        // projection has to be split identically.  Row-parallel keeps its bias
        // whole -- see shard_column_parallel_bias().
        if (desc.is_column_parallel) {
            shard_column_parallel_bias(matmul, full, slice);
        }
    }

    if (runtime_sliced != 0 && std::getenv("TP_PROF") != nullptr) {
        std::cerr << "[TP] Rank " << rank << ": " << runtime_sliced << " of " << plan.linears.size()
                  << " weights could not be pre-sliced and fall back to a runtime Slice"
                  << " (this dominates compile time)" << std::endl;
    }

    // ------------------------------------------------------------------
    // 2) Patch Reshape constants after q/k/v projections.
    //
    //    The Reshape converts [B,S,features] → [B,S,heads,head_dim].
    //    The shape constant is [0, 0, num_heads, head_dim] with special_zero.
    //    We update index 2 (the head count).
    // ------------------------------------------------------------------
    for (const auto& desc : plan.linears) {
        if (desc.role != ShardingPlan::LinearDesc::Q_PROJ &&
            desc.role != ShardingPlan::LinearDesc::K_PROJ &&
            desc.role != ShardingPlan::LinearDesc::V_PROJ)
            continue;

        auto it = name_map.find(desc.matmul_name);
        if (it == name_map.end())
            continue;
        auto matmul = it->second;

        int64_t new_heads = (desc.role == ShardingPlan::LinearDesc::Q_PROJ)
                                ? local_q_heads
                                : local_kv_heads;

        // Find the direct Reshape consumer of the projection output (the bias
        // Add, when the projection has one).
        for (const auto& target : projection_output(matmul).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (ov::is_type<ov::op::v1::Reshape>(consumer)) {
                patch_reshape_constant(consumer, /*shape_idx=*/2, new_heads);
            }
        }
    }

    // ------------------------------------------------------------------
    // 2b) Patch the grouped-query broadcast that expands KV heads up to the Q
    //     head count.
    //
    //     The broadcast has two shapes in the wild (multiply by a ones tensor,
    //     or a bidirectional Broadcast), both of which the shared matcher
    //     covers.  Its root is the Reshape that merges the expanded heads back
    //     into [B, num_heads, S, head_dim]; index 1 of its shape constant is
    //     the head count we have to localize.
    //
    //     PagedAttention has no such broadcast: it reads the kv head count off
    //     the cache and groups internally, so there is nothing to patch and
    //     nothing to demand.
    // ------------------------------------------------------------------
    if (plan.attention_backend == ShardingPlan::AttentionBackend::SDPA) {
        auto kv_bcst = ov::op::util::match_multi_query_bcst(ov::pass::pattern::any_input());
        ov::pass::pattern::Matcher matcher(std::get<0>(kv_bcst), "TPMultiQueryBcst");

        size_t patched = 0;
        for (const auto& op : cloned->get_ordered_ops()) {
            if (!ov::is_type<ov::op::v1::Reshape>(op) || !matcher.match(op->output(0)))
                continue;
            auto shape_const =
                ov::as_type_ptr<ov::op::v0::Constant>(op->input(1).get_source_output().get_node_shared_ptr());
            if (!shape_const)
                continue;
            auto shape_data = shape_const->cast_vector<int64_t>();
            if (shape_data.size() == 4 && shape_data[1] == static_cast<int64_t>(plan.num_heads)) {
                patch_reshape_constant(op, /*shape_idx=*/1, local_q_heads);
                ++patched;
            }
        }

        OPENVINO_ASSERT(plan.num_kv_heads == plan.num_heads || patched > 0,
                        "[TP_GPU] The model uses grouped-query attention (", plan.num_kv_heads,
                        " KV heads for ", plan.num_heads,
                        " Q heads) but no KV broadcast was found to re-shape");
    }

    // PagedAttention needs no head patching of its own.  The conversion wraps
    // it in reshapes that are entirely relative -- `[0, -1]` flattening the
    // operands, and `Concat([0], [1], [-1], ShapeOf(key)[-1])` restoring the
    // heads afterwards -- so once the projections are sharded those reshapes
    // already carry the local head count.  The op reads the kv head count off
    // the cache tensor bound at runtime, not off the graph.

    // ------------------------------------------------------------------
    // 2c) Patch the post-SDPA Reshape that flattens [B,S,heads,head_dim] back
    //     into hidden_size before o_proj.
    //
    //     Topology in Llama-style models:
    //         SDPA → Transpose → Reshape([0,0,hidden_size]) → o_proj
    //     (sometimes with a Convert in the chain).
    //
    //     After column-parallel sharding of q/k/v, the heads dim is local;
    //     the flattened output must be local_q_heads * head_dim, otherwise
    //     o_proj's MatMul shape inference fails (input dim != weight K).
    //
    //     Some exports build this Reshape's shape via ShapeOf+Concat
    //     (fully dynamic) — in that case input(1) is not a Constant and we
    //     simply skip; nothing to patch. Other exports (e.g. Llama-3.x-8B)
    //     bake a static Constant [0,0,hidden_size] which we patch in place.
    //
    //     We walk back from o_proj.input(0) through transparent ops only —
    //     scoping the scan to the o_proj producer chain — to avoid touching
    //     unrelated Reshapes that happen to contain the same value.
    // ------------------------------------------------------------------
    {
        const int64_t hidden = static_cast<int64_t>(plan.hidden_size);
        const int64_t local_hidden =
            local_q_heads * static_cast<int64_t>(plan.head_dim);

        for (const auto& desc : plan.linears) {
            if (desc.role != ShardingPlan::LinearDesc::O_PROJ)
                continue;

            auto it = name_map.find(desc.matmul_name);
            if (it == name_map.end())
                continue;
            auto matmul = it->second;

            // Walk back through transparent passthroughs (Convert) until we
            // reach the Reshape, or give up after a small bounded hop count.
            std::shared_ptr<ov::Node> src =
                matmul->input(0).get_source_output().get_node_shared_ptr();
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
            if (!shape_const)
                continue;  // dynamic shape (ShapeOf+Concat) — nothing to do

            auto data = shape_const->cast_vector<int64_t>();
            bool patched = false;
            for (auto& v : data) {
                if (v == hidden) {
                    v = local_hidden;
                    patched = true;
                }
            }
            if (!patched)
                continue;

            auto new_const = ov::op::v0::Constant::create(
                shape_const->get_element_type(), shape_const->get_shape(), data);
            src->input(1).replace_source_output(new_const->output(0));
        }
    }

    // ------------------------------------------------------------------
    // 3) Adjust KV cache Variable shapes.
    //
    //    ReadValue / Assign variables for KV cache have shape
    //    [batch, kv_heads, seq, head_dim].  Update dim[1] to local_kv_heads.
    // ------------------------------------------------------------------
    for (const auto& variable : collect_kv_cache_variables(*cloned, plan.num_kv_heads)) {
        auto info = variable->get_info();
        info.data_shape[1] = local_kv_heads;
        variable->update(info);
    }

    // ------------------------------------------------------------------
    // 3b) Patch the KV cache initialization subgraph.
    //
    //     ReadValue init input comes from:
    //       Constant(0.0) + Concat([batch, kv_heads, 0, head_dim])
    //         → Broadcast → ReadValue
    //
    //     Find Constants with value [num_kv_heads] that feed into a Concat
    //     whose output feeds a Broadcast that feeds a ReadValue.
    //     Replace the kv_heads constant with [local_kv_heads].
    //
    //     Note: every ReadValue in the model has its OWN init chain with its
    //     own kv_heads Constant node, even if all of them share the same
    //     binary offset in the IR.  We must patch every match — a single
    //     `goto` after the first hit leaves 63 of 64 init constants holding
    //     the original num_kv_heads on Llama-3.x-8B and fails Variable
    //     shape validation downstream.
    // ------------------------------------------------------------------
    {
        std::vector<std::shared_ptr<ov::op::v0::Constant>> kv_init_consts;
        for (const auto& op : cloned->get_ordered_ops()) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!c)
                continue;
            // Look for scalar-in-vector constant with value == num_kv_heads
            if (c->get_shape() != ov::Shape{1})
                continue;
            auto data = c->cast_vector<int64_t>();
            if (data.size() != 1 || data[0] != static_cast<int64_t>(plan.num_kv_heads))
                continue;

            // Confirm chain: Constant -> Concat -> Broadcast -> ReadValue.
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
            auto new_c = ov::op::v0::Constant::create(
                c->get_element_type(), c->get_shape(),
                std::vector<int64_t>{local_kv_heads});
            c->output(0).replace(new_c->output(0));
        }
    }

    // ------------------------------------------------------------------
    // 3c) Insert TPAllReduce after each row-parallel MatMul.
    //
    //     Row-parallel outputs (o_proj, down_proj) are partial sums across
    //     ranks and need AllReduce.  We insert an explicit TPAllReduce op
    //     between the MatMul output and its consumers, making the
    //     collective visible in the graph.  The GPU plugin executes
    //     TPAllReduce as an in-graph CPU primitive via the shared
    //     TPDeviceCoordinator object.
    // ------------------------------------------------------------------
    {
        uint32_t collective_id = 0;
        for (const auto& desc : plan.linears) {
            if (desc.is_column_parallel)
                continue;  // only row-parallel needs AllReduce

            auto it = name_map.find(desc.matmul_name);
            if (it == name_map.end())
                continue;
            auto matmul = it->second;

            auto ar = std::make_shared<ov::tp_gpu::op::TPAllReduce>(
                matmul->output(0),
                /*group_id=*/0,
                /*collective_id=*/collective_id++,
                /*rank=*/rank,
                /*world_size=*/tp_degree,
                /*reduce_kind=*/"sum");
            ar->set_friendly_name("tp_allreduce/" + desc.matmul_name);

            // Redirect all consumers of the MatMul to use the AllReduce output.
            auto targets = matmul->output(0).get_target_inputs();
            for (const auto& target : targets) {
                if (target.get_node() == ar.get())
                    continue;
                target.replace_source_output(ar->output(0));
            }
        }
    }

    // ------------------------------------------------------------------
    // 4) Validate — propagate shapes through the modified graph.
    // ------------------------------------------------------------------
    cloned->validate_nodes_and_infer_types();

    return cloned;
}

int GraphRewriter::count_collectives(const ShardingPlan& plan) {
    int count = 0;
    for (const auto& desc : plan.linears) {
        if (!desc.is_column_parallel)
            ++count;
    }
    return count;
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
