// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "pooling_inst.h"
#include "fully_connected_inst.h"
#include "resample_inst.h"
#include "strided_slice_inst.h"
#include "reorder_inst.h"
#include "concatenation_inst.h"
#include "read_value_inst.h"
#include "assign_inst.h"
#include "kv_cache_inst.h"
#include "stateless_kv_inst.h"
#include "condition_inst.h"
#include "loop_inst.h"
#include "input_layout_inst.h"
#include "lora_inst.h"
#include "mutable_data_inst.h"
#include "transformations/symbolic_transformations/utils.hpp"
#include <algorithm>
#include <optional>
#include <utility>
#include <vector>
#include <sstream>

namespace cldnn {
void program_helpers::reshape_deconvolution_weights(const std::vector<float> &deconv_weights,
    const int channels,
    const int kernel_width,
    const int kernel_height,
    const int scale_factor,
    std::vector<std::vector<std::vector<float> > >& subpixel_weights) {

    std::vector<std::vector<float> > weights(channels);

    int pad_zero_x = kernel_width % 2 == 0 ? 0 : 1;
    int pad_zero_y = kernel_height % 2 == 0 ? 0 : 1;

    // reshape 9x9 deconv weights, for example 32 9x9 deconv weights to 32 10x10 conv weights
    for (int f = 0; f < channels; ++f) {
        for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                int index = f * kernel_width * kernel_height + kernel_y * kernel_width + kernel_x;
                weights[f].push_back(deconv_weights[index]);
            }
            if (pad_zero_x == 1) {    // pad with zero on x axis
                weights[f].push_back(0.f);
            }
        }
        if (pad_zero_y == 1) {    // pad a line on y axis with zero
            for (int kernel_x = 0; kernel_x < kernel_width + pad_zero_x; ++kernel_x) {
                weights[f].push_back(0.f);
            }
        }
    }

    // reshape 32 10x10 weights to 4 32 5x5 weights
    for (int s = 0; s < scale_factor*scale_factor; ++s) {
        subpixel_weights[s].resize(channels);
    }

    const int kernel_sz = kernel_width + pad_zero_x;

    auto get_row_index = [](int index, const int kernel_sz)->int {
        bool isRowEven = (index / (kernel_sz)) % 2 == 0;
        bool isColEven = (index % 2) == 0;
        int kernel_num = isRowEven ? (isColEven ? 0 : 1) : isColEven ? 2 : 3;
        return kernel_num;
    };

    int feature_num = static_cast<int>(weights.size());
    for (int f = 0; f < feature_num; ++f) {
        for (int i = 0; i < static_cast<int>(weights[f].size()); ++i) {
            int row = get_row_index(i, kernel_sz);
            subpixel_weights[row][f].push_back(weights[f][i]);
        }
    }

    // dump the weights for the shuffled kernel
    int subpixel_conv_num = static_cast<int>(subpixel_weights.size());
    for (int s = 0; s < subpixel_conv_num; ++s) {
        for (int row = 0; row < static_cast<int>(subpixel_weights[s].size()); ++row) {
            std::reverse(std::begin(subpixel_weights[s][row]), std::end(subpixel_weights[s][row]));
        }
    }
}

static bool is_full_tensor(const layout& l) {
    return l.spatial(0) > 1 || l.spatial(1) > 1 || (l.get_spatial_rank() == 3 && l.spatial(2) > 1)
        || l.batch() > 1;
}

void onednn_eltwise_fusing_helpers::for_eltwise(
    const program_node& node, eltwise_mode mode,
    std::function<void(const program_node& p_node,
                    const fused_primitive_desc& desc)> func) {
    for (const auto& fo : node.get_fused_primitives()) {
        if (fo.is_type<eltwise>() && fo.typed_desc<eltwise>()->mode == mode) {
            func(node, fo);
        }
    }
}

static bool is_direct_ancestor(const program_node& child, const program_node& target) {
    // is_direct_ancestor function is added to detect pattern like below from get_add_fusing_type.
    // It is necessary to use onednn sum post operation in such case for better performance.
    // In such case, 'A' can have two connections.
    //   ┌───────┐
    //   │   A   │
    //   └──┬──┬─┘
    //      │  └────────────┐
    //      │           ┌───┴───┐
    //      │post_op    │   B   │
    //      │           └───┬───┘
    //      │  ┌────────────┘
    //   ┌──┴──┴─┐
    //   │   C   │
    //   └───────┘
    if (target.get_users().size() != 2)
        return false;

    // The walk follows dependency(0) only, so this bounds the length of the single-input chain
    // between the two nodes; the equality test precedes the hop, so a bound of N reaches a target
    // at most N - 1 hops away.
    //
    // An attention residual sits 6 hops up. Walking dependency(0) from the projection that carries
    // the post-op, on a graph dump of a multi-head attention block:
    //
    //     0  the output projection            <- carries the fused eltwise sum
    //     1  reshape                          <- heads folded back into the feature axis
    //     2  the attention op
    //     3  reshape                          <- feature axis split into heads
    //     4  the QKV matmul
    //     5  the norm
    //     6  the block input                  <- the residual root, two users
    //
    // 7 is therefore the smallest bound that reaches it, and the bound is kept at that minimum
    // deliberately. Note the two reshapes: it is easy to count this chain as 4 hops from the
    // operator list alone and conclude a bound of 5 or 6 is enough, and both leave every
    // attention residual demoted.
    //
    // Raising it beyond the minimum is not a compile-time cost -- the walk is a bounded pointer
    // chase. It is a trade. What the result decides is add_fusing_type::sum versus binary_add, and
    // both are post-ops on the same onednn primitive: sum lets onednn accumulate into the added
    // tensor's buffer, at the price of disabling implicit concat on that predecessor
    // (prepare_buffer_fusing), inhibiting reorder elimination in two passes (layout_optimizer and
    // remove_redundant_reorders) and blocking buffer sharing (basic_memory_dependencies). So a
    // longer chain should only be admitted against a workload that is measured to want it.
    static constexpr int max_ancestor_walk_depth = 7;

    const auto* iter = &child;
    for (int i = 0; i < max_ancestor_walk_depth; i++) {
        if (iter == &target)
            return true;
        if (iter->get_dependencies().empty())
            break;
        iter = &iter->get_dependency(0);
    }
    return false;
}

bool program_helpers::is_buffer_view_of_input0(const program_node& n, view_resolution mode) {
    if (n.get_dependencies().empty()) {
        return false;
    }
    // owners or stop points of the walk
    if (n.is_type<concatenation>() || n.is_type<read_value>() || n.is_type<input_layout>() || n.is_type<data>() ||
        n.is_type<mutable_data>()) {
        return false;
    }
    if (n.can_be_optimized() || n.is_runtime_skippable()) {
        return true;
    }
    if (mode == view_resolution::settled_only) {
        return false;
    }
    // Candidates that become views only AFTER the oneDNN attribute freeze (they mirror the passes that run after
    // add_onednn_optimization_attributes: mark_runtime_skippable_nodes and the post-phase remove_redundant_reorders).
    if (n.is_dynamic() || n.has_fused_primitives()) {
        return false;
    }
    if (n.is_type<resample>()) {
        return n.get_input_layout(0) == n.get_output_layout();
    }
    if (n.is_type<strided_slice>()) {
        const auto& in = n.get_input_layout(0);
        const auto& out = n.get_output_layout();
        return in.format == out.format && in.data_type == out.data_type;
    }
    if (n.is_type<reorder>()) {
        // The post-freeze remove_redundant_reorders rewrites reorders freely (chain shrinking, padding merges,
        // duplicate merges), so any plain single-input reorder may end up as a view of its input.
        return n.get_dependencies().size() == 1 && !n.as<reorder>().get_primitive()->has_surface_input();
    }
    return false;
}

const program_node* program_helpers::in_place_concat_holder_of(const program_node& n) {
    for (const auto* u : n.get_users()) {
        if (u->is_type<concatenation>() && u->can_be_optimized()) {
            return u;
        }
    }
    return nullptr;
}

bool program_helpers::is_or_becomes_output(const program_node& n) {
    if (n.is_output()) {
        return true;
    }
    for (const auto* u : n.get_users()) {
        if (u->is_type<mutable_data>() && is_or_becomes_output(*u)) {
            return true;
        }
    }
    return false;
}

bool program_helpers::is_external_buffer_root(const program_node& n) {
    // The list also covers condition and loop: they re-point their output buffers on every execution (condition
    // postprocess_output_memory, loop back-edge rebinding), so a writer dst bound at construction time goes stale.
    return n.is_type<condition>() || n.is_type<loop>() || n.is_type<input_layout>() || n.is_type<data>() ||
           n.is_type<mutable_data>() || n.is_type<read_value>() || n.is_type<assign>() || n.is_type<kv_cache>() ||
           n.is_type<stateless_kv>() || n.is_constant();
}

program_helpers::buffer_owner program_helpers::resolve_buffer_owner(const program_node& n, view_resolution mode, bool collect_path) {
    buffer_owner res;
    // Backward view steps followed by forward slot steps; concat is never a view and settled views cannot be concat
    // predecessors, so the walk is finite. The bound is a safety net only.
    const size_t bound = n.get_program().get_processing_order().size();
    size_t steps = 0;
    const program_node* cur = &n;
    int32_t cur_port = 0;
    if (collect_path) {
        res.path.push_back(cur);
    }
    res.path_has_output = is_or_becomes_output(*cur);
    while (true) {
        if (const auto* holder = in_place_concat_holder_of(*cur)) {
            cur = holder;  // slot step takes precedence
            cur_port = 0;
        } else if (is_buffer_view_of_input0(*cur, mode)) {
            const auto dep = cur->get_dependency_with_port(0);
            cur = dep.first;
            cur_port = dep.second;
        } else {
            break;
        }
        OPENVINO_ASSERT(++steps <= bound, "[GPU] buffer owner walk does not terminate from node ", n.id());
        if (collect_path) {
            res.path.push_back(cur);
        }
        res.path_has_output = res.path_has_output || is_or_becomes_output(*cur);
    }
    res.owner = cur;
    res.owner_port = cur_port;
    return res;
}

namespace {
// F10 helper (allocation-free, recursive over concat cascades): true if any predecessor slot of the in-place concat
// `holder` has a reader other than the holder (or is a network output), or is itself a concat holding such slots.
bool concat_slots_have_other_readers(const program_node& holder) {
    for (const auto& d : holder.get_dependencies()) {
        const auto* s = d.first;
        if (s->is_output()) {
            return true;
        }
        for (const auto* u : s->get_users()) {
            if (u != &holder) {
                return true;
            }
        }
        if (s->is_type<concatenation>() && s->can_be_optimized() && concat_slots_have_other_readers(*s)) {
            return true;
        }
    }
    return false;
}
}  // namespace

add_fusing_type onednn_eltwise_fusing_helpers::get_add_fusing_type(
    const program_node& p_node, const fused_primitive_desc& desc, add_fusing_query q) {
    if (!desc.is_type<eltwise>()) {
        return add_fusing_type::not_supported;
    }
    if (desc.typed_desc<eltwise>()->mode != eltwise_mode::sum) {
        return add_fusing_type::not_supported;
    }
    if (!desc.has_outer_dep()) {
        return add_fusing_type::not_supported;
    }
    // prepare_primitive_fusing only fuses a two-input eltwise with p_node as one input, so exactly one
    // dependency arrives here and the single index below is enough.
    OPENVINO_ASSERT(desc.deps.size() == 1,
                    "[GPU] A fused eltwise sum must bring exactly one dependency, got ",
                    desc.deps.size(),
                    " on node ",
                    p_node.id());
    auto& dep_node = p_node.get_dependency(desc.outer_dep_start_idx);
    auto p_layout = p_node.get_output_layout();
    auto d_layout = dep_node.get_output_layout();

    if (p_node.is_dynamic() || dep_node.is_dynamic()) {
        return add_fusing_type::not_supported;
    }

    // dep_node's buffer owner, resolved at most once and only when a clause needs it; the F11, F10 and F6 clauses below
    // share it. Exact queries can run per inference (static can_be_optimized resample users re-bind their sum dst via
    // rebind_onednn_reuse_optimized_dst_if_needed -> get_reused_eltwmem_idx), so the walk is allocation-free:
    // collect_path stays false and F6 reads the path_has_output summary instead of the path itself.
    std::optional<program_helpers::buffer_owner> dep_owner_cache;
    auto dep_owner = [&]() -> const program_helpers::buffer_owner& {
        if (!dep_owner_cache) {
            dep_owner_cache = program_helpers::resolve_buffer_owner(
                dep_node, program_helpers::view_resolution::include_post_freeze_candidates, /*collect_path=*/false);
        }
        return *dep_owner_cache;
    };

    // onednn accumulates a sum post-op into dep_node's buffer, so p_node must not also read it;
    // is_direct_ancestor answers yes for an own input too
    auto p_node_consumes_dep_node = [&]() {
        // has_outer_dep() above rejects a negative index.
        const auto outer_dep_start_idx = static_cast<size_t>(desc.outer_dep_start_idx);
        for (size_t i = 0; i < outer_dep_start_idx; i++) {
            const auto& in = p_node.get_dependency(i);
            if (&in == &dep_node)
                return true;
            // F11: p_node must not read dep_node's buffer through a view on its own inputs either. Views that only
            // appear after the freeze (static no-op resample/strided_slice, reorders) are included, so the freeze
            // and the post-freeze callers resolve identically.
            if (q == add_fusing_query::exact) {
                if (program_helpers::resolve_buffer_owner(
                        in, program_helpers::view_resolution::include_post_freeze_candidates).owner == dep_owner().owner)
                    return true;
            }
        }
        return false;
    };

    // F10: the dst would overwrite in-place concat slots that have readers other than the concat. Conservative
    // Pre-0 placeholder for the Phase 1a war_safe rule (it also rejects readers that run before p_node); any
    // relaxation must be applied to this predicate at the freeze, because F5 can only downgrade later.
    // NOTE: until F5, the frozen value and this live value can still diverge for reasons outside F10/F11
    // (users count, padding, the identity test after reorder removal).
    auto overwrites_shared_concat_slot = [&]() {
        if (q != add_fusing_query::exact) {
            return false;
        }
        const auto* owner = dep_owner().owner;
        return owner->is_type<concatenation>() && owner->can_be_optimized() && concat_slots_have_other_readers(*owner);
    };

    // F6/F13: the dst of the sum is dep_node's buffer, so it must be neither an externally owned buffer (input_layout is
    // replaced by set_data without notifying the writer; data, mutable_data and variable buffers are not ours to
    // overwrite) nor a buffer that is or becomes user-visible. Exact only: guard_superset stays a superset of exact.
    auto writes_external_or_visible_buffer = [&]() {
        if (q != add_fusing_query::exact) {
            return false;
        }
        if (program_helpers::is_or_becomes_output(p_node)) {
            return true;
        }
        const auto& bo = dep_owner();
        if (program_helpers::is_external_buffer_root(*bo.owner)) {
            return true;
        }
        // any node from dep to owner (concat holders and view steps included) is or becomes a network output
        return bo.path_has_output;
    };

    if (is_full_tensor(p_layout) && is_full_tensor(d_layout)) {
        if (data_type_traits::size_of(p_layout.data_type) == data_type_traits::size_of(d_layout.data_type)
            && p_layout.format == d_layout.format && p_layout.get_tensor() == d_layout.get_tensor()
            && p_layout.data_padding == d_layout.data_padding
            && (dep_node.get_users().size() == 1 || is_direct_ancestor(p_node, dep_node))
            && !dep_node.is_constant()
            && !p_node.is_type<pooling>()
            && !p_node.is_output()
            // Only matters for guard_superset: in exact mode any input_layout owner is rejected by F6 below.
            && (!dep_node.is_type<input_layout>() || dep_node.get_users().size() <= 1)
            && !p_node_consumes_dep_node()
            && !overwrites_shared_concat_slot()
            && !writes_external_or_visible_buffer()) {
            return add_fusing_type::sum;
        }
        if (p_layout.get_tensor() == d_layout.get_tensor()) {
            return add_fusing_type::binary_per_tensor;
        }
    }

    return add_fusing_type::binary_per_oc;
}

// F17 M3b/M3c helper (allocation-free, recursion only over the forward view chain of `cur`): true if the buffer of `cur` is, or is
// viewed by, a network output (M3b), or is a slot of a concat that is or may become in-place (M3c). The walk advances ONLY to users
// that are settled views of input 0 of exactly `cur` (C3): other users own their buffers and do not share dst.
bool onednn_eltwise_fusing_helpers::forward_view_chain_is_visible_or_concat_slot(const program_node& cur) {
    if (program_helpers::is_or_becomes_output(cur)) {
        return true;
    }
    for (const auto* u : cur.get_users()) {
        if (u->is_type<concatenation>() && (u->can_be_optimized() || u->is_runtime_skippable())) {
            return true;
        }
    }
    for (const auto* u : cur.get_users()) {
        if (!u->get_dependencies().empty() && &u->get_dependency(0) == &cur &&
            program_helpers::is_buffer_view_of_input0(*u, program_helpers::view_resolution::settled_only) &&
            forward_view_chain_is_visible_or_concat_slot(*u)) {
            return true;
        }
    }
    return false;
}

// F17 M8 / F5: the type recorded at the oneDNN attribute freeze (and possibly downgraded by the F5 re-freeze) for the fused op whose
// outer dependency is `outer_dep_idx`. Keyed by mem_dep, not by position (the list is reordered/pruned by post-op optimization).
bool onednn_eltwise_fusing_helpers::is_frozen_inplace_post_op(const program_node& node, int32_t outer_dep_idx, eltwise_mode mode) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    for (const auto& e : node.get_fused_primitives_onednn()) {
        if (e.mem_dep != static_cast<size_t>(outer_dep_idx)) {
            continue;
        }
        if (mode == eltwise_mode::sum) {
            if (e.op_type == onednn_post_op_type::sum || e.op_type == onednn_post_op_type::optimized_sum)
                return true;
        } else if (e.op_type == onednn_post_op_type::binary_mul_inplace) {
            return true;
        }
    }
#else
    (void)node;
    (void)outer_dep_idx;
    (void)mode;
#endif
    return false;
}

onednn_eltwise_fusing_helpers::sum_downgrade_reason onednn_eltwise_fusing_helpers::frozen_sum_downgrade_reason(const program_node& n,
                                                                                                             int32_t mem_dep) {
    // The single decision function of F5: the re-freeze pass, the cache-import check and the debug assertions all call it.
    if (n.get_preferred_impl_type() != impl_types::onednn) {
        return sum_downgrade_reason::preferred_not_onednn;
    }
    if (n.is_in_shape_of_subgraph()) {
        return sum_downgrade_reason::in_shape_of_subgraph;
    }
    if (!n.get_selected_impl() || !n.get_selected_impl()->is_onednn()) {
        return sum_downgrade_reason::impl_not_onednn;
    }
    if (mem_dep < 0 || static_cast<size_t>(mem_dep) >= n.get_dependencies().size() || n.get_dependency_with_port(mem_dep).second != 0) {
        return sum_downgrade_reason::dep_port_not_zero;
    }
    // the same predicate the unbound dst binding uses: the writer (or a view chain starting at it) must not be a concat slot
    if (forward_view_chain_is_visible_or_concat_slot(n)) {
        return sum_downgrade_reason::writer_is_concat_slot_or_visible;
    }
    const fused_primitive_desc* sum_op = nullptr;
    for (const auto& fused_op : n.get_fused_primitives()) {
        if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1 && fused_op.has_outer_dep() && fused_op.outer_dep_start_idx == mem_dep &&
            fused_op.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
            sum_op = &fused_op;
            break;
        }
    }
    if (!sum_op || get_add_fusing_type(n, *sum_op) != add_fusing_type::sum) {
        return sum_downgrade_reason::live_not_sum;
    }
    return sum_downgrade_reason::none;
}

// binary_mul_inplace: oneDNN matmul post-op whose src1 may alias dst. Unlike sum the alias is
// optional (plain binary_mul otherwise), so this only checks the buffers are interchangeable:
// FC only, dynamic via ov::Symbol, exact layout match, single user.
bool onednn_eltwise_fusing_helpers::can_use_mul_inplace(const program_node& p_node, const fused_primitive_desc& desc) {
    if (!desc.is_type<eltwise>() || desc.typed_desc<eltwise>()->mode != eltwise_mode::prod || !desc.has_outer_dep()
        || !p_node.is_type<fully_connected>()) {
        return false;
    }

    auto& dep_node = p_node.get_dependency(desc.outer_dep_start_idx);
    auto p_layout = p_node.get_output_layout();
    auto d_layout = dep_node.get_output_layout();

    if (p_layout.is_dynamic() || d_layout.is_dynamic()) {
        if (p_layout.data_type != d_layout.data_type || p_layout.format != d_layout.format
            || p_layout.data_padding != d_layout.data_padding) {
            return false;
        }
        const auto& p_pshape = p_layout.get_partial_shape();
        const auto& d_pshape = d_layout.get_partial_shape();
        if (p_pshape.rank().is_dynamic() || d_pshape.rank().is_dynamic() || p_pshape.size() != d_pshape.size()) {
            return false;
        }
        for (size_t i = 0; i < p_pshape.size(); ++i) {
            if (!ov::symbol::util::dims_are_equal(p_pshape[i], d_pshape[i])) {
                return false;
            }
        }
    } else if (p_layout != d_layout || !is_full_tensor(p_layout)) {
        return false;
    }

    // M0 (pre-F17 conditions)
    if (dep_node.get_users().size() != 1 || dep_node.is_constant() || p_node.is_output()) {
        return false;
    }

    // F17. The alias is only safe if dep's buffer is exclusively dep's own, read by p_node alone through this one input, and
    // invisible outside the network. Every rule below is a pure structural query: it runs at the freeze, at get_reused_eltwmem_idx
    // callers (also per inference) and must give the same answer to all of them.
    // M6: dst is bound to output port 0 of the dependency
    if (p_node.get_dependency_with_port(desc.outer_dep_start_idx).second != 0) {
        return false;
    }
    // M5: p_node reads dep through another input as well (users == 1 means that is the only way it can read it twice)
    size_t reads_of_dep = 0;
    for (const auto& d : p_node.get_dependencies()) {
        if (d.first == &dep_node) {
            reads_of_dep++;
        }
    }
    if (reads_of_dep != 1) {
        return false;
    }
    // M1: dep must own its buffer. settled_only on purpose: for a mul, "accepted at the freeze, rejected live" is a harmless
    // downgrade and the reverse is stopped by M8; callers all run after the graph passes, when the flags are stable.
    if (program_helpers::is_buffer_view_of_input0(dep_node, program_helpers::view_resolution::settled_only) ||
        program_helpers::in_place_concat_holder_of(dep_node) != nullptr || dep_node.is_type<lora>()) {
        return false;
    }
    // M2: input_layout (replaced by set_data), data, mutable_data, variables, condition, loop
    if (program_helpers::is_external_buffer_root(dep_node)) {
        return false;
    }
    // M3: neither dep nor p_node may be or become user-visible (past_outputs bypasses the restriction carry-over for the latter)
    if (program_helpers::is_or_becomes_output(dep_node) || program_helpers::is_or_becomes_output(p_node)) {
        return false;
    }
    // M3b/M3c: p_node itself, or a view chain starting at it, is visible or a concat slot (dst would write the concat's buffer)
    if (onednn_eltwise_fusing_helpers::forward_view_chain_is_visible_or_concat_slot(p_node)) {
        return false;
    }
    // M4: dep is a concat that is or may become in-place
    if (dep_node.is_type<concatenation>() && (dep_node.can_be_optimized() || dep_node.is_runtime_skippable())) {
        return false;
    }
    // M7: shape_of subgraphs and CPU impls (dynamic nodes that may land on CPU are a Phase 0.5 gap)
    if (dep_node.is_in_shape_of_subgraph() ||
        (dep_node.get_selected_impl() ? dep_node.get_selected_impl()->is_cpu() : dep_node.get_preferred_impl_type() == impl_types::cpu)) {
        return false;
    }
    return true;
}

int32_t onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(const program_node& node) {
    if (node.get_preferred_impl_type() == impl_types::onednn) {
        for (const auto& fused_op : node.get_fused_primitives()) {
            if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                auto mode = fused_op.typed_desc<eltwise>()->mode;
                bool reuse_eligible = (mode == eltwise_mode::sum && get_add_fusing_type(node, fused_op) == add_fusing_type::sum) ||
                                      (mode == eltwise_mode::prod && can_use_mul_inplace(node, fused_op));
                if (!reuse_eligible)
                    continue;
                if (!fused_op.has_outer_dep())
                    continue;
#ifdef ENABLE_ONEDNN_FOR_GPU
                // F17 M8: only ever downgrade. A static node's attrs were frozen once; a live "eligible" answer that disagrees with the
                // frozen post-op type of THIS fused op (e.g. frozen binary_mul / binary_add) must not alias dst, because oneDNN was not
                // told that src1 and dst coincide. Dynamic nodes rebuild their attrs from the same live predicate in update_impl; the
                // impl actually used is checked at the binding points (primitive_inst::reused_eltw_alias_allowed).
                if (!node.is_dynamic() && !is_frozen_inplace_post_op(node, fused_op.outer_dep_start_idx, mode))
                    continue;
#endif
                return fused_op.outer_dep_start_idx;
            }
        }
    }
    return -1;   // if -1, no reused input memory
}
}  // namespace cldnn
