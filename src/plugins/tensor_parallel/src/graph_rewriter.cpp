// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_rewriter.hpp"

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <unordered_map>

#include "tensor_parallel/op/tp_all_reduce.hpp"
#include "tensor_parallel/tp_coordination.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/variable.hpp"

namespace ov {
namespace tp {

namespace {

/// Safe type name comparison (get_type_name() returns const char*).
inline bool type_is(const std::shared_ptr<ov::Node>& node, const char* name) {
    return std::strcmp(node->get_type_name(), name) == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// analyze()
// ---------------------------------------------------------------------------

ShardingPlan GraphRewriter::analyze(const std::shared_ptr<const ov::Model>& model) {
    ShardingPlan plan;

    // Regex: matches "layers.{N}.{submodule}.{proj}/...MatMul" or similar
    // We key on the friendly name containing "layers.N" and a known proj suffix.
    std::regex layer_re(R"(layers\.(\d+))");

    for (const auto& op : model->get_ordered_ops()) {
        if (!type_is(op, "MatMul"))
            continue;

        const auto& name = op->get_friendly_name();
        std::smatch m;
        if (!std::regex_search(name, m, layer_re))
            continue;

        int layer_idx = std::stoi(m[1].str());

        ShardingPlan::LinearDesc desc;
        desc.matmul_name = name;
        desc.layer_idx = layer_idx;

        if (name.find("self_attn.q_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::Q_PROJ;
            desc.is_column_parallel = true;
        } else if (name.find("self_attn.k_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::K_PROJ;
            desc.is_column_parallel = true;
        } else if (name.find("self_attn.v_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::V_PROJ;
            desc.is_column_parallel = true;
        } else if (name.find("self_attn.o_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::O_PROJ;
            desc.is_column_parallel = false;
        } else if (name.find("mlp.gate_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::GATE_PROJ;
            desc.is_column_parallel = true;
        } else if (name.find("mlp.up_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::UP_PROJ;
            desc.is_column_parallel = true;
        } else if (name.find("mlp.down_proj") != std::string::npos) {
            desc.role = ShardingPlan::LinearDesc::DOWN_PROJ;
            desc.is_column_parallel = false;
        } else {
            continue;
        }

        plan.linears.push_back(desc);
        plan.num_layers = std::max(plan.num_layers, layer_idx + 1);
    }

    // Extract model dimensions from layer 0 weights and reshape constants.
    for (const auto& op : model->get_ordered_ops()) {
        if (!type_is(op, "MatMul"))
            continue;
        const auto& name = op->get_friendly_name();

        // --- hidden_size from q_proj weight [out, in] with transpose_b ---
        if (name.find("layers.0.self_attn.q_proj") != std::string::npos) {
            auto ws = op->input(1).get_partial_shape();
            if (ws.rank().is_static() && ws.rank().get_length() == 2 && ws[0].is_static()) {
                plan.hidden_size = static_cast<int>(ws[0].get_length());
            }
            // Infer num_heads and head_dim from the Reshape consumer
            for (const auto& target : op->output(0).get_target_inputs()) {
                auto consumer = target.get_node()->shared_from_this();
                if (type_is(consumer, "Reshape")) {
                    auto shape_node = consumer->input(1).get_source_output().get_node()->shared_from_this();
                    if (auto sc = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_node)) {
                        auto data = sc->cast_vector<int64_t>();
                        if (data.size() >= 4) {
                            plan.num_heads = static_cast<int>(data[2]);
                            plan.head_dim = static_cast<int>(data[3]);
                        }
                    }
                }
            }
        }

        // --- num_kv_heads from k_proj weight ---
        if (name.find("layers.0.self_attn.k_proj") != std::string::npos && plan.head_dim > 0) {
            auto ws = op->input(1).get_partial_shape();
            if (ws.rank().is_static() && ws[0].is_static()) {
                plan.num_kv_heads = static_cast<int>(ws[0].get_length()) / plan.head_dim;
            }
        }

        // --- intermediate_size from gate_proj weight ---
        if (name.find("layers.0.mlp.gate_proj") != std::string::npos) {
            auto ws = op->input(1).get_partial_shape();
            if (ws.rank().is_static() && ws[0].is_static()) {
                plan.intermediate_size = static_cast<int>(ws[0].get_length());
            }
        }
    }

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

/// Map an axis backwards through a Reshape by matching cumulative element
/// counts before the sliced dimension.
///
/// When the output dimension spans multiple consecutive input dimensions
/// (e.g. Reshape [N, G, D] → [N, G*D]), returns the outermost of those
/// input dimensions so long as the slice boundaries are guaranteed to
/// align with the inner-dim granularity.
///
/// out_start/out_end: the slice boundaries on the output tensor.
/// Returns {input_axis, input_full_size, input_start, input_end},
/// or {-1, 0, 0, 0} if the mapping fails.
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
    auto out_ps = reshape->get_output_partial_shape(0);
    auto in_ps = reshape->input(0).get_source_output().get_partial_shape();
    if (!in_ps.rank().is_static() || !out_ps.rank().is_static())
        return {-1, 0, 0, 0};

    size_t out_before = 1;
    for (int64_t d = 0; d < output_axis && d < out_ps.rank().get_length(); ++d) {
        if (!out_ps[d].is_static()) return {-1, 0, 0, 0};
        out_before *= out_ps[d].get_length();
    }

    size_t cum = 1;
    for (int64_t d = 0; d < in_ps.rank().get_length(); ++d) {
        if (!in_ps[d].is_static()) return {-1, 0, 0, 0};
        if (cum == out_before) {
            int64_t in_dim = static_cast<int64_t>(in_ps[d].get_length());
            // Case 1: exact 1-to-1 dimension match
            if (in_dim == full_size) {
                return {d, in_dim, out_start, out_end};
            }
            // Case 2: output dim spans multiple input dims (e.g. [G, D] → [G*D])
            // Compute the product of trailing dims that make up the output dim.
            int64_t trailing = 1;
            for (int64_t dd = d + 1; dd < in_ps.rank().get_length(); ++dd) {
                if (!in_ps[dd].is_static()) return {-1, 0, 0, 0};
                trailing *= static_cast<int64_t>(in_ps[dd].get_length());
                if (in_dim * trailing == full_size) {
                    // Found: output dim = in_shape[d] * in_shape[d+1] * ... * in_shape[dd]
                    // Slice boundaries must be aligned with `trailing` granularity.
                    if (out_start % trailing == 0 && out_end % trailing == 0) {
                        return {d, in_dim, out_start / trailing, out_end / trailing};
                    }
                    return {-1, 0, 0, 0};  // not aligned
                }
            }
            return {-1, 0, 0, 0};
        }
        cum *= in_ps[d].get_length();
    }
    return {-1, 0, 0, 0};
}

/// Read-only check: can the decompression chain be pre-sliced?
/// Returns true if every Reshape in the chain can map the axis cleanly.
bool can_pre_slice_chain(ov::Output<ov::Node> output,
                          int64_t axis,
                          int64_t full_size,
                          int64_t start,
                          int64_t end) {
    auto node = output.get_node_shared_ptr();

    if (std::dynamic_pointer_cast<ov::op::v0::Constant>(node))
        return true;

    auto tn = std::string(node->get_type_name());

    if (tn == "Convert") {
        return can_pre_slice_chain(node->input(0).get_source_output(), axis, full_size, start, end);
    } else if (tn == "Subtract" || tn == "Multiply" || tn == "Add") {
        return can_pre_slice_chain(node->input(0).get_source_output(), axis, full_size, start, end) &&
               can_pre_slice_chain(node->input(1).get_source_output(), axis, full_size, start, end);
    } else if (tn == "Reshape") {
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

    if (auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
        auto shape = c->get_shape();
        if (axis >= 0 && axis < static_cast<int64_t>(shape.size()) &&
            static_cast<int64_t>(shape[axis]) == full_size) {
            auto sliced = pre_slice_constant(c, axis, start, end);
            sliced->set_friendly_name(c->get_friendly_name() + "_shard");
            input.replace_source_output(sliced->output(0));
        }
        return;
    }

    auto tn = std::string(node->get_type_name());

    if (tn == "Convert") {
        pre_slice_chain(node->input(0), axis, full_size, start, end);
    } else if (tn == "Subtract" || tn == "Multiply" || tn == "Add") {
        pre_slice_chain(node->input(0), axis, full_size, start, end);
        pre_slice_chain(node->input(1), axis, full_size, start, end);
    } else if (tn == "Reshape") {
        auto m = map_reshape_axis(node, axis, full_size, start, end);
        pre_slice_chain(node->input(0), m.axis, m.full_size, m.start, m.end);

        // Patch the Reshape shape constant.
        // When the output dim maps 1:1 to an input dim, patch at `axis`.
        // When it spans multiple input dims, patch at `m.axis` in the
        // Reshape's shape vector (the outermost input dim that changed).
        auto sc = std::dynamic_pointer_cast<ov::op::v0::Constant>(
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

/// Replace a Reshape's shape constant, updating one element.
/// `shape_idx` is the index within the shape vector to overwrite.
void patch_reshape_constant(const std::shared_ptr<ov::Node>& reshape,
                            size_t shape_idx,
                            int64_t new_value) {
    auto shape_node = reshape->input(1).get_source_output().get_node()->shared_from_this();
    auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_node);
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

}  // namespace

// ---------------------------------------------------------------------------
// rewrite()
// ---------------------------------------------------------------------------

std::shared_ptr<ov::Model> GraphRewriter::rewrite(const std::shared_ptr<const ov::Model>& model,
                                                  const ShardingPlan& plan,
                                                  uint32_t rank,
                                                  uint32_t tp_degree,
                                                  const std::shared_ptr<TPCoordination>& coordination) {
    OPENVINO_ASSERT(tp_degree >= 2, "[TP] tp_degree must be >= 2");
    OPENVINO_ASSERT(plan.num_heads % tp_degree == 0,
                    "[TP] num_heads (", plan.num_heads, ") not divisible by tp_degree (", tp_degree, ")");
    OPENVINO_ASSERT(plan.num_kv_heads % tp_degree == 0,
                    "[TP] num_kv_heads (", plan.num_kv_heads, ") not divisible by tp_degree (", tp_degree, ")");
    OPENVINO_ASSERT(plan.intermediate_size % tp_degree == 0,
                    "[TP] intermediate_size (", plan.intermediate_size,
                    ") not divisible by tp_degree (", tp_degree, ")");

    auto cloned = model->clone();

    int64_t local_q_heads = plan.num_heads / static_cast<int64_t>(tp_degree);
    int64_t local_kv_heads = plan.num_kv_heads / static_cast<int64_t>(tp_degree);

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
    for (const auto& desc : plan.linears) {
        auto it = name_map.find(desc.matmul_name);
        if (it == name_map.end())
            continue;
        auto matmul = it->second;

        auto weight_output = matmul->input(1).get_source_output();
        auto ws = weight_output.get_partial_shape();
        if (!ws.rank().is_static() || ws.rank().get_length() != 2)
            continue;

        int64_t N = ws[0].get_length();  // output features (transpose_b)
        int64_t K = ws[1].get_length();  // input features

        // Column-parallel: shard axis 0 (output dim)
        // Row-parallel:    shard axis 1 (input dim)
        int64_t axis = desc.is_column_parallel ? 0 : 1;
        int64_t full = desc.is_column_parallel ? N : K;

        int64_t shard = full / static_cast<int64_t>(tp_degree);
        int64_t start_idx = static_cast<int64_t>(rank) * shard;
        int64_t end_idx = (rank == tp_degree - 1) ? full : (start_idx + shard);

        if (can_pre_slice_chain(weight_output, axis, full, start_idx, end_idx)) {
            pre_slice_chain(matmul->input(1), axis, full, start_idx, end_idx);
        } else {
            auto sliced = insert_weight_slice(weight_output, axis, start_idx, end_idx);
            matmul->input(1).replace_source_output(sliced);
        }
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

        // Find the direct Reshape consumer of the MatMul output.
        for (const auto& target : matmul->output(0).get_target_inputs()) {
            auto consumer = target.get_node()->shared_from_this();
            if (type_is(consumer, "Reshape")) {
                patch_reshape_constant(consumer, /*shape_idx=*/2, new_heads);
            }
        }
    }

    // ------------------------------------------------------------------
    // 2b) Patch GQA expansion Reshapes.
    //
    //     After KV Broadcast expansion, a Reshape merges
    //     [B, kv_heads, GQA_ratio, S, head_dim] → [B, num_heads, S, head_dim].
    //     Shape constant is [0, num_heads, -1, head_dim].
    //     We update index 1 to local_q_heads.
    // ------------------------------------------------------------------
    for (const auto& op : cloned->get_ordered_ops()) {
        if (!type_is(op, "Reshape"))
            continue;
        // Input[0] must be a Broadcast (the GQA expansion).
        auto src = op->input(0).get_source_output().get_node()->shared_from_this();
        if (!type_is(src, "Broadcast"))
            continue;
        // Check the shape constant at index 1.
        auto shape_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
            op->input(1).get_source_output().get_node()->shared_from_this());
        if (!shape_node)
            continue;
        auto shape_data = shape_node->cast_vector<int64_t>();
        if (shape_data.size() == 4 &&
            shape_data[1] == static_cast<int64_t>(plan.num_heads)) {
            patch_reshape_constant(op, /*shape_idx=*/1, local_q_heads);
        }
    }

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
            for (int hops = 0; hops < 4 && src && !type_is(src, "Reshape"); ++hops) {
                if (type_is(src, "Convert")) {
                    src = src->input(0).get_source_output().get_node_shared_ptr();
                } else {
                    src.reset();
                    break;
                }
            }
            if (!src || !type_is(src, "Reshape"))
                continue;

            auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(
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
    // Collect unique variables (ReadValue and Assign share the same Variable).
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> var_map;
    for (const auto& op : cloned->get_ordered_ops()) {
        std::shared_ptr<ov::op::util::Variable> variable;
        if (auto rv = std::dynamic_pointer_cast<ov::op::v6::ReadValue>(op)) {
            variable = rv->get_variable();
        } else if (auto assign = std::dynamic_pointer_cast<ov::op::v6::Assign>(op)) {
            variable = assign->get_variable();
        }
        if (!variable)
            continue;

        auto var_id = variable->get_info().variable_id;
        if (var_id.find("past_key_values") == std::string::npos &&
            var_id.find("key") == std::string::npos &&
            var_id.find("value") == std::string::npos)
            continue;

        if (var_map.count(var_id))
            continue;
        var_map[var_id] = variable;
    }

    for (auto& [var_id, variable] : var_map) {
        auto info = variable->get_info();
        auto& shape = info.data_shape;
        if (shape.rank().is_static() && shape.rank().get_length() == 4 &&
            shape[1].is_static() &&
            shape[1].get_length() == static_cast<int64_t>(plan.num_kv_heads)) {
            shape[1] = local_kv_heads;
            variable->update(info);
        }
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
            auto c = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
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
                if (!type_is(concat, "Concat"))
                    continue;
                for (const auto& ct : concat->output(0).get_target_inputs()) {
                    auto broadcast = ct.get_node()->shared_from_this();
                    if (!type_is(broadcast, "Broadcast"))
                        continue;
                    for (const auto& bt : broadcast->output(0).get_target_inputs()) {
                        if (type_is(bt.get_node()->shared_from_this(), "ReadValue")) {
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
    //     TPCoordination object.
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

            auto ar = std::make_shared<ov::op::tp::TPAllReduce>(
                matmul->output(0),
                /*group_id=*/0,
                /*collective_id=*/collective_id++,
                /*world_size=*/tp_degree,
                /*reduce_kind=*/"sum");
            ar->set_friendly_name("tp_allreduce/" + desc.matmul_name);

            // Store coordination context in rt_info for the GPU plugin's op factory.
            ar->get_rt_info()["tp_coordination"] = coordination;
            ar->get_rt_info()["tp_rank"] = static_cast<int64_t>(rank);

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

}  // namespace tp
}  // namespace ov
