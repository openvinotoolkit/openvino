// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_sliding_window_kvcache.hpp"

#include <algorithm>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>

#include "../logging.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"

namespace {

// Matches e.g. "past_key_values.3.key" / "past_key_values.12.value" -> layer index 3 / 12.
const std::regex& kv_param_regex() {
    static const std::regex re(R"(past_key_values\.(\d+)\.(?:key|value))");
    return re;
}

// Matches e.g. "...layers.5.self_attn..." -> layer index 5. Reused from the same
// convention as embedding/prepare_embedding_model.cpp's AddKVCacheNodes matcher.
const std::regex& layer_id_regex() {
    static const std::regex re(R"(layers\.(\d+)\.self_attn)");
    return re;
}

bool try_parse_layer_idx(const std::string& text, const std::regex& re, size_t& out_idx) {
    std::smatch m;
    if (!std::regex_search(text, m, re)) {
        return false;
    }
    out_idx = static_cast<size_t>(std::stoul(m[1].str()));
    return true;
}

}  // namespace

namespace ov::npuw {

PatchSlidingWindowKVCache::PatchSlidingWindowKVCache(uint32_t window_size,
                                                     std::vector<bool> layer_is_sliding,
                                                     uint32_t kvcache_size,
                                                     uint32_t input_size,
                                                     const KVAxesPosition& kv_axes_position,
                                                     bool trim_attention_mask)
    : m_window_size(window_size),
      m_layer_is_sliding(std::move(layer_is_sliding)),
      m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position),
      m_trim_attention_mask(trim_attention_mask) {}

bool PatchSlidingWindowKVCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (m_window_size == 0 || m_layer_is_sliding.empty()) {
        LOG_INFO("[SWA] Sliding Window Attention is not configured, skipping " << model->get_friendly_name());
        return false;
    }

    // `new_kv_total` is the resulting post-concat K/V length seen by a sliding-window
    // layer's SDPA (past + current input). The sliding-window causal constraint is
    // per-query-row: the EARLIEST new query row in this call (absolute position P,
    // i.e. tokens already stored before this call) needs past tokens
    // [P-window_size+1, P-1], i.e. `window_size - 1` of history - independent of how
    // many additional new rows (`m_input_size`) follow it in the same call. So past
    // capacity must track `window_size` directly, NOT `window_size - input_size`
    // (subtracting input_size here was a correctness bug: with input_size == window_size,
    // e.g. chunked prefill, it collapsed past to 0, silently dropping all history the
    // first row of the chunk should have attended to).
    // Degenerate case: if window_size >= kvcache_size, sliding effectively never kicks
    // in within the model's overall budget, so behave like a regular (non-SWA) layer:
    // past = kvcache_size - input_size.
    const uint32_t new_past_u32 = (m_window_size < m_kvcache_size)
                                      ? m_window_size
                                      : (m_input_size < m_kvcache_size ? (m_kvcache_size - m_input_size) : 0u);
    const int64_t new_past = static_cast<int64_t>(new_past_u32);
    const int64_t new_kv_total = new_past + static_cast<int64_t>(m_input_size);
    bool changed = false;

    // Step 0: BEFORE touching any Parameter shape, snapshot the *current* (fully static, still
    // uniform-`m_kvcache_size`) value of EVERY self_attn layer's shape/bound-carrying input that
    // depends on the current KV length - both the GQA/repeat_kv `expand` lowering's target shape
    // (Concat(past,new) -> Unsqueeze -> Broadcast -> Reshape -> SDPA) AND the per-layer causal-mask
    // trim (HF's `causal_mask[..., : key_states.shape[-2]]`, lowered to `aten::slice` ->
    // `opset8::Slice(mask, begin, end, step, axis)`, whose `end` bound is exactly this same
    // current-KV-length value) - sliding *and* full-attention layers alike. This must happen
    // strictly first, and must cover ALL layers, for two reasons:
    //
    //  1) `model->reshape()`/`validate_nodes_and_infer_types()` set-and-validate atomically, so
    //     any stale downstream shape needs to already be fixed *before* the single validation
    //     call at the end of this function - there's no chance to patch things "in between".
    //
    //  2) Some exporters/compilers common-subexpression-eliminate (CSE) the shape-computation
    //     subgraph feeding this input across MULTIPLE layers that happen to be structurally
    //     identical at trace time (all layers, sliding or full-attention, have the SAME uniform
    //     kvcache_size before this pass runs). Confirmed on a real Gemma model dump: one
    //     ShapeOf(Concat(past,new))->Gather->Concat chain, whose ShapeOf happened to be sourced
    //     from one PARTICULAR layer's own Concat, fed the Broadcast of SEVERAL other layers too
    //     (a mix of sliding AND full-attention ones). Only patching the sliding layers' consumer
    //     edges (and leaving full-attention layers alone, assuming they "don't need to change")
    //     is still wrong: once the representative source layer's Parameter is resized in Step 1,
    //     EVERY OTHER layer still wired to that shared node - including full-attention layers we
    //     never intended to touch - inherits the new (wrong-for-them) value too. This is exactly
    //     what caused a real crash where a *full-attention* layer's Broadcast target shape
    //     unexpectedly showed the *sliding* size, and then (same root cause) a *full-attention*
    //     layer's own causal-mask Slice's `end` bound too. So every layer's shape/bound input
    //     touching this pattern must be decoupled from the (possibly shared) upstream
    //     computation, regardless of that layer's own sliding status.
    //
    // The robust fix is to depend on neither of the above: right now, *before* any Parameter is
    // reshaped, the whole model is still uniformly static, so `ov::util::get_constant_from_source()`
    // can fold ANY such input - a plain Constant, or a live (possibly shared) ShapeOf-based chain -
    // down to its current concrete value. We snapshot that value now, then later (Step 2, after
    // Parameter shapes have changed) build a brand-new, per-node *private* Constant - for sliding
    // layers with the stale `m_kvcache_size` entries corrected to `new_kv_total`, for
    // full-attention layers with the SAME original value, just privatized - and hook it up via
    // `replace_source_output()`. This never mutates or reuses the original (possibly shared)
    // upstream node, so it works uniformly regardless of how the shape is represented or how many
    // layers currently reuse the same shape-computation subgraph.
    struct ShapeInputSnapshot {
        std::shared_ptr<ov::Node> consumer;
        size_t input_index;
        size_t layer_idx;
        bool layer_is_sliding;
        std::vector<int64_t> original_values;
    };
    std::vector<ShapeInputSnapshot> shape_snapshots;
    // Nodes that ARE a per-layer causal-mask Slice (`causal_mask[..., begin:end]`) whose bound was
    // found and will be privatized/corrected below. Tracked by NODE POINTER, not by the layer_idx
    // parsed from the node's own friendly name: this Slice can ITSELF be CSE-shared across
    // multiple layers' SDPA mask inputs (confirmed via a real crash - see Step 3's comment), so a
    // single node whose name says "layer 0" may in fact be the actual mask source feeding many
    // OTHER layers' SDPA too. For any SDPA whose mask input traces back to one of these nodes, the
    // mask is ALREADY made consistent with the new KV length by this very mechanism - Step 3
    // further down must NOT also trim it (that would double-trim it).
    std::unordered_set<const ov::Node*> privatized_mask_slice_nodes;
    for (const auto& node : model->get_ordered_ops()) {
        // Reshape/Broadcast: target-shape input is input(1). Slice (the per-layer causal-mask
        // trim `causal_mask[..., begin:end]`): both `begin` (input 1) and `end` (input 2) are
        // guarded defensively, though in practice only `end` carries the current-KV-length value.
        std::vector<size_t> candidate_input_indices;
        const bool is_mask_slice = ov::is_type<ov::op::v8::Slice>(node);
        if (ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v3::Broadcast>(node)) {
            candidate_input_indices = {1};
        } else if (is_mask_slice) {
            candidate_input_indices = {1, 2};
        } else {
            continue;
        }
        size_t layer_idx = 0;
        if (!try_parse_layer_idx(node->get_friendly_name(), layer_id_regex(), layer_idx)) {
            continue;
        }
        if (layer_idx >= m_layer_is_sliding.size()) {
            continue;
        }
        for (const size_t input_idx : candidate_input_indices) {
            if (input_idx >= node->get_input_size()) {
                continue;
            }
            auto folded = ov::util::get_constant_from_source(node->input_value(input_idx));
            if (!folded) {
                LOG_WARN("[SWA] Layer " << layer_idx << ": " << node->get_type_name() << " '"
                                        << node->get_friendly_name() << "' input(" << input_idx
                                        << ") is not constant-foldable, cannot verify/patch it.");
                continue;
            }
            auto values = folded->cast_vector<int64_t>();
            const auto kv = static_cast<int64_t>(m_kvcache_size);
            const bool has_kvcache_dim =
                std::any_of(values.begin(), values.end(), [kv](int64_t v) { return v == kv || v == -kv; });
            if (!has_kvcache_dim) {
                // Not a KV-size-dependent shape/bound (e.g. unrelated input in the same
                // self_attn block) - nothing to guard here.
                continue;
            }
            shape_snapshots.push_back(
                ShapeInputSnapshot{node, input_idx, layer_idx, m_layer_is_sliding[layer_idx], std::move(values)});
            if (is_mask_slice) {
                privatized_mask_slice_nodes.insert(node.get());
            }
        }
    }

    // Step 1: shrink past_key_values Parameter shapes for sliding-window layers.
    //
    // NOTE: we deliberately do NOT call `model->reshape()` here - see the rationale in Step 0's
    // comment above. Instead we set the new Parameter shape directly
    // (`Parameter::set_partial_shape`, which does not trigger any validation), and defer the
    // single `validate_nodes_and_infer_types()` call to the very end of this function, once Step 2
    // has also fixed up the Broadcast/Reshape/Slice shape-bound inputs snapshotted in Step 0.
    size_t num_params_reshaped = 0;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        size_t layer_idx = 0;
        if (!try_parse_layer_idx(name, kv_param_regex(), layer_idx)) {
            continue;
        }
        if (layer_idx >= m_layer_is_sliding.size() || !m_layer_is_sliding[layer_idx]) {
            continue;
        }
        const auto& pshape = input.get_partial_shape();
        if (pshape.rank().is_dynamic() || m_kv_axes_position.seq_len >= pshape.size() ||
            !pshape[m_kv_axes_position.seq_len].is_static()) {
            LOG_WARN("[SWA] Layer " << layer_idx << ": past KV parameter '" << name
                                    << "' has a non-static seq_len axis, skipping reshape. Was ReshapeToStatic "
                                       "applied before PatchSlidingWindowKVCache?");
            continue;
        }
        const int64_t old_past = pshape[m_kv_axes_position.seq_len].get_length();
        if (new_past == old_past) {
            continue;
        }
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        if (!param) {
            LOG_WARN("[SWA] Layer " << layer_idx << ": past KV input '" << name
                                    << "' is not backed by a Parameter node, skipping.");
            continue;
        }
        ov::PartialShape new_shape = pshape;
        new_shape[m_kv_axes_position.seq_len] = new_past;
        param->set_partial_shape(new_shape);
        changed = true;
        ++num_params_reshaped;
        LOG_INFO("[SWA] Layer " << layer_idx << ": past KV '" << name << "' seq_len " << old_past << " -> "
                                 << new_past << " (window=" << m_window_size << ", kvcache_size=" << m_kvcache_size
                                 << ", input_size=" << m_input_size << ", post-concat total=" << new_kv_total
                                 << ")");
    }
    if (num_params_reshaped > 0) {
        LOG_INFO("[SWA] Reshaped " << num_params_reshaped << " past_key_values parameter(s) in '"
                                   << model->get_friendly_name() << "' for sliding-window layers.");
    } else {
        LOG_INFO("[SWA] No past_key_values parameters required a reshape in '" << model->get_friendly_name()
                                                                                << "'.");
    }

    // Step 2: apply the fix for every shape/bound input snapshotted in Step 0 (Broadcast/Reshape
    // target shapes, and Slice begin/end bounds for the per-layer causal-mask trim). For each,
    // take its ORIGINAL (pre-patch) value and rebuild it as a brand-new, PRIVATE Constant via
    // `replace_source_output()` - never mutating the original (possibly shared) upstream node in
    // place:
    //   - sliding-window layers: entries equal to `m_kvcache_size` (or `-m_kvcache_size`, for
    //     Slice bounds expressed as a negative/from-the-end offset) are corrected to
    //     `new_kv_total` (signed to match).
    //   - full-attention layers: entries are kept at their original value - but STILL rebuilt as a
    //     private Constant, purely to sever any (possibly shared) live upstream dependency that a
    //     sibling sliding layer might later invalidate (see Step 0's comment for why this matters
    //     even when the value itself doesn't change).
    {
        // Step 2a: for sliding-window layers' mask-Slice nodes specifically, the naive "correct the
        // `end` bound constant" approach above is not enough. HF's `causal_mask[..., :new_kv_total]`
        // lowers to a Slice with begin=0 (front-aligned), but the SWA runtime KV-write logic keeps the
        // physical past_key_values buffer LEFT-aligned, holding the most recent tokens in chronological
        // order (see the "KV buffer invariant" - oldest kept token at column 0, newest at the last
        // column). So after this call, the buffer holds exactly ONE contiguous window of ABSOLUTE
        // positions: `[chunk_start + input_size - new_kv_total, chunk_start + input_size)`, where
        // `chunk_start` is this call's first query position (varies across calls that reuse the SAME
        // compiled graph: different chunks of the same chunked-prefill model, or different steps served
        // by the same generate kv_size variant). A single Slice with a FIXED begin (whether 0, as HF
        // emits, or a fixed tail offset) can only ever be correct for ONE such call.
        //
        // Fix: replace the whole mask-Slice node with a Gather whose index tensor is
        // `Range(0, new_kv_total, 1) + begin`, `begin` computed at runtime from `position_ids[0]`.
        // NOTE: an earlier version of this fix used `Slice(begin, end)` with runtime bounds, pinned
        // to a static shape via a `Reshape(special_zero=true)`. That crashed NPUW's online
        // partitioning ("identifyUniques" pass -> "to_shape was called on a dynamic shape"): the
        // model's own Parameter shapes are updated in Step 1 but downstream shape propagation is
        // deferred to this function's own final `validate_nodes_and_infer_types()` call, so at the
        // time this pass runs some non-sliced axes of the mask tensor are still legitimately dynamic
        // (to be resolved by a LATER pipeline stage) - `special_zero` just copies that dynamism
        // through, and the *sliced* axis's static pin doesn't help those OTHER axes. Gather doesn't
        // have this problem: its output shape at the gathered axis comes purely from the INDEX
        // tensor's OWN (always-static, `new_kv_total`-long) shape, and every other axis is passed
        // through exactly as-is (dynamic or not) with no separate "pin" step required.
        std::shared_ptr<ov::Node> position_ids_node;
        for (const auto& input : model->inputs()) {
            if (input.get_any_name() == "position_ids") {
                position_ids_node = input.get_node_shared_ptr();
                break;
            }
        }
        const auto pos_ids_pshape = position_ids_node ? position_ids_node->get_output_partial_shape(0) : ov::PartialShape{};
        const bool position_ids_usable =
            position_ids_node && pos_ids_pshape.rank().is_static() && pos_ids_pshape.size() == 2;

        // Lazily built, shared across layers. Kept as an explicit [1]-shaped (not fully-squeezed to a
        // rank-0 scalar) tensor: the NPU/vpux compiler's own type inference for a Squeeze reducing this
        // to a true scalar disagrees with its declared result type - Subtract/Maximum/Add below all
        // broadcast a [1]-shaped operand just as well, so there's no need to squeeze it.
        std::shared_ptr<ov::Node> chunk_start_scalar;
        auto get_chunk_start = [&]() {
            if (!chunk_start_scalar) {
                auto idx0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
                chunk_start_scalar = std::make_shared<ov::op::v8::Gather>(position_ids_node, idx0, axis1);
            }
            return chunk_start_scalar;
        };

        std::unordered_set<const ov::Node*> dynamically_reselected;
        size_t reselected_count = 0;
        if (position_ids_usable && m_window_size < m_kvcache_size) {
            for (const auto& snapshot : shape_snapshots) {
                if (!snapshot.layer_is_sliding || !privatized_mask_slice_nodes.count(snapshot.consumer.get()) ||
                    dynamically_reselected.count(snapshot.consumer.get())) {
                    continue;
                }
                auto slice_node = snapshot.consumer;
                auto axis_const = ov::util::get_constant_from_source(slice_node->input_value(4));
                if (!axis_const) {
                    LOG_WARN("[SWA] Layer " << snapshot.layer_idx << ": mask Slice '"
                                            << slice_node->get_friendly_name()
                                            << "' has a non-constant-foldable axis, cannot dynamically reselect its "
                                               "columns - falling back to the static-bound correction below.");
                    continue;
                }
                const int64_t axis = axis_const->cast_vector<int64_t>().at(0);

                const auto data_pshape = slice_node->get_input_partial_shape(0);
                if (data_pshape.rank().is_dynamic()) {
                    LOG_WARN("[SWA] Layer " << snapshot.layer_idx << ": mask Slice '"
                                            << slice_node->get_friendly_name()
                                            << "' has a dynamic-rank data input, cannot dynamically reselect its "
                                               "columns - falling back to the static-bound correction below.");
                    continue;
                }
                const int64_t rank = data_pshape.rank().get_length();
                const int64_t norm_axis = axis < 0 ? axis + rank : axis;
                if (norm_axis < 0 || norm_axis >= rank) {
                    LOG_WARN("[SWA] Layer " << snapshot.layer_idx << ": mask Slice '"
                                            << slice_node->get_friendly_name() << "' has an out-of-range axis "
                                            << axis << " for rank " << rank
                                            << " - falling back to the static-bound correction below.");
                    continue;
                }
                dynamically_reselected.insert(slice_node.get());

                // begin_raw = chunk_start - window_size (UNCLAMPED, may be negative - this is the
                // mathematically "true" start of the physical "past" region's absolute-position
                // window). begin_1d = max(0, begin_raw) is only used to keep the Gather's own index
                // values non-negative/valid; the possible discrepancy it introduces (see below) is
                // compensated for explicitly, instead of silently accepted as before.
                auto window_size_const =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(m_window_size)});
                auto begin_raw = std::make_shared<ov::op::v1::Subtract>(get_chunk_start(), window_size_const);
                auto zero_scalar = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto begin_1d = std::make_shared<ov::op::v1::Maximum>(begin_raw, zero_scalar);  // shape [1]

                // The physical buffer is [past (window_size wide), new (input_size wide)] - reselect
                // each part with its OWN index expression instead of one combined
                // `Range(0, new_kv_total) + begin`:
                //
                //  - "new" part: always `Range(0, input_size) + chunk_start`, UNCLAMPED. `chunk_start`
                //    (from position_ids) is never negative, so this is always valid, and it always
                //    picks exactly this call's own query positions - the columns holding this call's
                //    real, just-written K/V.
                //
                //  - "past" part: `Range(0, window_size) + begin_1d`, clamped as before (Gather can't
                //    take negative indices). When `chunk_start >= window_size` (the steady-state case)
                //    `begin_raw` is already >= 0, so this is identical to the previous single-Gather
                //    behavior. But when `chunk_start < window_size` (only possible for the very FIRST
                //    call using this compiled graph, e.g. the first chunk of chunked prefill, since
                //    position_ids only grows afterwards), `begin_raw` is negative and clamping to 0
                //    shifts the "past" part's gathered columns to alias the SAME absolute positions
                //    the "new" part just picked - silently making not-yet-existing "past" slots look
                //    like valid, visible duplicates of the current chunk's own tokens (instead of the
                //    intended "no real history yet" - invisible). Since the physical past_key_values
                //    buffer content at those slots is never actually written for such a call, this
                //    physically-nonexistent-but-visible aliasing let attention draw on garbage/stale
                //    KV data - the discrepancy this whole block corrects.
                //
                //    Fix: explicitly force any "past" slot whose UNCLAMPED absolute position
                //    (`begin_raw + local_p`) is still negative back to invisible (-inf), regardless of
                //    which (borrowed/aliased) column the clamped Gather happened to read.
                auto idx_new_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto idx_new_stop =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(m_input_size)});
                auto idx_step = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
                auto idx_new_range =
                    std::make_shared<ov::op::v4::Range>(idx_new_start, idx_new_stop, idx_step, ov::element::i64);
                auto new_indices =
                    std::make_shared<ov::op::v1::Add>(idx_new_range, get_chunk_start());  // shape [input_size]

                auto idx_past_start = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
                auto idx_past_stop =
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(m_window_size)});
                auto idx_past_range =
                    std::make_shared<ov::op::v4::Range>(idx_past_start, idx_past_stop, idx_step, ov::element::i64);
                auto past_indices = std::make_shared<ov::op::v1::Add>(idx_past_range, begin_1d);  // shape [window_size]
                // Same per-slot offsets, but against the UNCLAMPED begin - negative entries mark
                // "past" slots that don't correspond to any real history yet.
                auto past_raw_pos = std::make_shared<ov::op::v1::Add>(idx_past_range, begin_raw);  // shape [window_size]
                auto invalid_past =
                    std::make_shared<ov::op::v1::Less>(past_raw_pos, zero_scalar);  // shape [window_size], BOOL

                auto axis_1d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
                auto new_gathered =
                    std::make_shared<ov::op::v8::Gather>(slice_node->input_value(0), new_indices, axis_1d);
                auto past_gathered =
                    std::make_shared<ov::op::v8::Gather>(slice_node->input_value(0), past_indices, axis_1d);

                // Broadcast `invalid_past` ([window_size]) to the mask's rank, with `window_size` at
                // `norm_axis` and `1` everywhere else, so it can Select elementwise against
                // `past_gathered` (whose other axes may still be dynamic at this point - the reshape
                // target itself is fully static, computed purely from `rank`/`norm_axis`/`window_size`).
                std::vector<int64_t> invalid_shape(static_cast<size_t>(rank), 1);
                invalid_shape[static_cast<size_t>(norm_axis)] = static_cast<int64_t>(m_window_size);
                auto invalid_shape_const = ov::op::v0::Constant::create(ov::element::i64,
                                                                        ov::Shape{invalid_shape.size()},
                                                                        invalid_shape);
                auto invalid_past_reshaped =
                    std::make_shared<ov::op::v1::Reshape>(invalid_past, invalid_shape_const, false);

                const auto data_elem_type = slice_node->get_input_element_type(0);
                auto neg_inf_const =
                    ov::op::v0::Constant::create(data_elem_type, ov::Shape{}, {-std::numeric_limits<float>::max()});
                auto past_corrected =
                    std::make_shared<ov::op::v1::Select>(invalid_past_reshaped, neg_inf_const, past_gathered);

                auto gathered = std::make_shared<ov::op::v0::Concat>(
                    ov::OutputVector{past_corrected, new_gathered},
                    axis);
                gathered->set_friendly_name(slice_node->get_friendly_name() + "/swa_dynamic_reselect");

                // Keep Step 3's "already handled" guard working: it looks up an SDPA's mask source
                // node in `privatized_mask_slice_nodes` to skip double-trimming. Since consumers now
                // point at `gathered` instead of the original Slice, register it too.
                privatized_mask_slice_nodes.insert(gathered.get());

                auto target_inputs = slice_node->output(0).get_target_inputs();
                for (auto&& input : target_inputs) {
                    input.replace_source_output(gathered);
                }
                changed = true;
                ++reselected_count;
                LOG_INFO("[SWA] Layer " << snapshot.layer_idx << ": replaced mask Slice '"
                                        << slice_node->get_friendly_name()
                                        << "' with a position_ids-anchored dynamic-begin Gather (window="
                                        << m_window_size << ", input_size=" << m_input_size
                                        << ", total_width=" << m_kvcache_size << ").");
            }
        }
        if (reselected_count > 0) {
            LOG_INFO("[SWA] Dynamically reselected " << reselected_count
                                                      << " sliding-layer mask Slice node(s) in '"
                                                      << model->get_friendly_name() << "' using position_ids.");
        }

        // Step 2b: the original constant-correction path, for everything NOT handled by Step 2a above
        // (Broadcast/Reshape shape-bound inputs, full-attention-layer mask-Slice bounds, and
        // sliding-layer mask-Slice bounds in the degenerate `window_size >= kvcache_size` case or when
        // `position_ids` wasn't usable).
        size_t patched_count = 0;
        size_t privatized_count = 0;
        for (const auto& snapshot : shape_snapshots) {
            if (dynamically_reselected.count(snapshot.consumer.get())) {
                continue;  // node fully replaced above; its old begin/end constants are now dead code.
            }
            auto values = snapshot.original_values;
            const int64_t kv = static_cast<int64_t>(m_kvcache_size);
            const int64_t corrected = snapshot.layer_is_sliding ? new_kv_total : kv;
            bool value_changed = false;
            for (auto& v : values) {
                if (v == kv && v != corrected) {
                    v = corrected;
                    value_changed = true;
                } else if (v == -kv && v != -corrected) {
                    v = -corrected;
                    value_changed = true;
                }
            }
            auto new_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
            new_const->set_friendly_name(snapshot.consumer->get_friendly_name() + "/swa_shape_patched_" +
                                          std::to_string(snapshot.input_index));
            snapshot.consumer->input(snapshot.input_index).replace_source_output(new_const);
            changed = true;
            ++privatized_count;
            if (value_changed) {
                ++patched_count;
                LOG_INFO("[SWA] Layer " << snapshot.layer_idx << ": patched " << snapshot.consumer->get_type_name()
                                        << " '" << snapshot.consumer->get_friendly_name() << "' input("
                                        << snapshot.input_index << ") constant " << m_kvcache_size << " -> "
                                        << new_kv_total);
            }
        }
        if (privatized_count > 0) {
            LOG_INFO("[SWA] Privatized " << privatized_count
                                         << " Broadcast/Reshape/Slice shape-bound input(s) (" << patched_count
                                         << " value-corrected, " << (privatized_count - patched_count)
                                         << " guarded-only) feeding self_attn SDPA node(s) in '"
                                         << model->get_friendly_name() << "'.");
        }
    }

    // Step 3 (generate model only): trim the SDPA attention-mask input to the same window, for
    // architectures where the mask reaching SDPA is NOT already per-layer trimmed to the current
    // KV length by the model itself.
    //
    // IMPORTANT: this must be SKIPPED for any SDPA whose mask input traces back to a node in
    // `privatized_mask_slice_nodes` (Step 0/2's causal-mask Slice privatization). Confirmed via a
    // real crash: for architectures that DO lower `causal_mask[..., :key_states.shape[-2]]` to a
    // per-layer `opset8::Slice`, Step 0/2 already corrects that Slice's own `end` bound to
    // `new_kv_total`, so the SDPA mask input is already the right width. Step 3 running on TOP of
    // that (still) used a STALE `mask_source.get_partial_shape()` - queried BEFORE the deferred
    // final `validate_nodes_and_infer_types()` - so it saw the OLD (pre-Step-2) width and inserted
    // an ADDITIONAL Slice on top, double-trimming the mask down to a wrong, too-narrow width
    // (observed: 898 instead of the correct new_kv_total).
    //
    // A SECOND, MORE SUBTLE layer to this same bug (also confirmed via a real crash): the mask
    // Slice node itself can be CSE-shared across MULTIPLE layers (exactly like the Broadcast/
    // Reshape sharing described in Step 0) - e.g. only ONE Slice node, whose friendly name happens
    // to say "layer 0", is in fact the actual mask source for 24 OTHER sliding layers' SDPA too.
    // So the guard here CANNOT be keyed by "the layer_idx parsed from the Slice node's own name"
    // (that only matches the one layer whose name the shared node happens to carry) - it must be
    // keyed by the ACTUAL node identity of `mask_source.get_node()`, checked against every node we
    // privatized in Step 0, regardless of which layer's name that node carries.
    if (m_trim_attention_mask) {
        std::unordered_map<ov::Node*, ov::Output<ov::Node>> mask_slice_cache;
        size_t sdpa_count = 0;
        size_t patched_count = 0;
        for (const auto& node : model->get_ordered_ops()) {
            auto sdpa = std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(node);
            if (!sdpa) {
                continue;
            }
            ++sdpa_count;
            size_t layer_idx = 0;
            if (!try_parse_layer_idx(sdpa->get_friendly_name(), layer_id_regex(), layer_idx)) {
                LOG_INFO("[SWA] SDPA node '" << sdpa->get_friendly_name()
                                             << "' has no parsable layer index, skipping mask trim.");
                continue;
            }
            if (layer_idx >= m_layer_is_sliding.size() || !m_layer_is_sliding[layer_idx]) {
                continue;
            }
            static constexpr size_t kMaskInputIdx = 3;  // Q=0, K=1, V=2, mask=3 (see attention.cpp SDPA_Inputs)
            if (sdpa->get_input_size() <= kMaskInputIdx) {
                LOG_INFO("[SWA] SDPA node '" << sdpa->get_friendly_name() << "' (layer " << layer_idx
                                             << ") has no attention-mask input, skipping mask trim.");
                continue;
            }
            auto mask_input = sdpa->input(kMaskInputIdx);
            const auto mask_source = mask_input.get_source_output();
            if (privatized_mask_slice_nodes.count(mask_source.get_node()) > 0) {
                LOG_INFO("[SWA] SDPA node '"
                        << sdpa->get_friendly_name() << "' (layer " << layer_idx << ") mask source '"
                        << mask_source.get_node()->get_friendly_name()
                        << "' is already trimmed to the current KV length via Step 2's causal-mask Slice "
                           "privatization - skipping redundant mask trim.");
                continue;
            }
            const auto mask_pshape = mask_source.get_partial_shape();
            if (mask_pshape.rank().is_dynamic() || mask_pshape.size() == 0 ||
                !mask_pshape[mask_pshape.size() - 1].is_static()) {
                LOG_WARN("[SWA] SDPA node '" << sdpa->get_friendly_name() << "' (layer " << layer_idx
                                            << ") has a dynamic mask shape, cannot trim.");
                continue;
            }
            const size_t last_axis = mask_pshape.size() - 1;
            const int64_t old_width = mask_pshape[last_axis].get_length();
            if (new_kv_total >= old_width) {
                // Mask is already narrow enough (e.g. window >= kvcache_size for this variant).
                continue;
            }

            ov::Output<ov::Node> sliced;
            auto cache_it = mask_slice_cache.find(mask_source.get_node());
            if (cache_it != mask_slice_cache.end()) {
                sliced = cache_it->second;
            } else {
                auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {old_width - new_kv_total});
                auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {old_width});
                auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(last_axis)});
                auto slice = std::make_shared<ov::op::v8::Slice>(mask_source, begin, end, step, axis);
                slice->set_friendly_name(mask_source.get_node()->get_friendly_name() + "/swa_mask_slice");
                sliced = slice->output(0);
                mask_slice_cache.emplace(mask_source.get_node(), sliced);
                LOG_INFO("[SWA] Inserted shared mask Slice in '"
                        << model->get_friendly_name() << "': width " << old_width << " -> " << new_kv_total
                        << " (source node: " << mask_source.get_node()->get_friendly_name() << ")");
            }
            mask_input.replace_source_output(sliced);
            ++patched_count;
            changed = true;
        }
        LOG_INFO("[SWA] '" << model->get_friendly_name() << "': scanned " << sdpa_count
                            << " SDPA node(s), patched mask input on " << patched_count
                            << " sliding-layer SDPA node(s) using " << mask_slice_cache.size()
                            << " unique Slice node(s).");
    }

    if (changed) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

}  // namespace ov::npuw
