// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "patch_sliding_window_kvcache.hpp"

#include <algorithm>
#include <regex>

#include "../logging.hpp"
#include "../util.hpp"
#include "detect_causal_mask.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"

namespace {

// SDPA input layout: query=0, key=1, value=2, [attn_mask=3, [scale=4, [sink=5]]] - see
// openvino/op/scaled_dot_product_attention.hpp.
constexpr size_t kMaskInputIdx = 3;

void set_mask_param_name(const std::shared_ptr<ov::op::v0::Parameter>& param, const std::string& name) {
    param->set_friendly_name(name);
    param->get_output_tensor(0).set_names({name});
}

// Matches e.g. "past_key_values.3.key" / "past_key_values.12.value" -> layer index 3 / 12.
const std::regex& kv_param_regex() {
    static const std::regex re(R"(past_key_values\.(\d+)\.(?:key|value))");
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
                                                     const KVAxesPosition& kv_axes_position)
    : m_window_size(window_size),
      m_layer_is_sliding(std::move(layer_is_sliding)),
      m_kvcache_size(kvcache_size),
      m_input_size(input_size),
      m_kv_axes_position(kv_axes_position) {}

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
    // Special case: a FULL one-shot call (`input_size >= kvcache_size`, i.e. regular,
    // non-chunked prefill processing the entire prompt in a single call) has no separate
    // past buffer at all - the original graph never concatenates a real past KV here (it's
    // reshaped to 0-length upstream and typically folded away), and windowing is fully
    // expressed by the mask CONTENT within the input itself. Forcing a nonzero past here
    // would shrink/grow the past_key_values Parameter's declared shape without a matching
    // change to the (already-disconnected) K/V producer, desyncing it from the
    // `sliding_window_attention_mask` width set below and failing shape validation.
    // Degenerate case: if window_size >= kvcache_size, sliding effectively never kicks
    // in within the model's overall budget, so behave like a regular (non-SWA) layer:
    // past = kvcache_size - input_size.
    const uint32_t new_past_u32 = (m_input_size >= m_kvcache_size)
                                      ? 0u
                                      : (m_window_size < m_kvcache_size
                                             ? m_window_size
                                             : (m_input_size < m_kvcache_size ? (m_kvcache_size - m_input_size)
                                                                              : 0u));
    const int64_t new_past = static_cast<int64_t>(new_past_u32);
    const int64_t new_kv_total = new_past + static_cast<int64_t>(m_input_size);
    bool changed = false;

    // Step 0: externalize every SLIDING-WINDOW self_attn SDPA's mask input to a single shared
    // `sliding_window_attention_mask` Parameter. Only sliding layers need this: Step 1 below
    // physically SHRINKS sliding layers' past_key_values buffer to `window_size`, which breaks
    // the shape of whatever native mask subgraph the exporter produced for them, so their mask
    // must be host-filled against the new, smaller shape instead. Full-attention/causal SDPAs are
    // intentionally left completely untouched here - their past_key_values is never resized by
    // this pass, so their native in-graph mask representation (whether an `is_causal=true`
    // attribute or an explicit mask subgraph) stays perfectly shape-valid on its own and does not
    // need externalizing. This also avoids the cost of host-constructing and copying a
    // full-`m_kvcache_size`-wide global mask tensor on every inference call.
    // Must run BEFORE the past_key_values shrink below (that logic assumes the graph is still
    // uniformly `m_kvcache_size`-wide at this point).
    {
        std::shared_ptr<ov::op::v0::Parameter> sliding_mask_param;
        size_t num_cut = 0;
        for (const auto& node : model->get_ordered_ops()) {
            auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(node);
            if (!sdpa) {
                continue;
            }
            bool is_sliding = false;
            const auto& rt_info = sdpa->get_rt_info();
            const auto it = rt_info.find(NPUW_SDPA_MASK_RT_KEY);
            if (it != rt_info.end()) {
                is_sliding = it->second.as<int64_t>() >= 0;
            }
            if (!is_sliding) {
                // Full-attention/causal SDPA - leave its mask input (or is_causal attribute)
                // exactly as the exporter produced it.
                continue;
            }
            // Every SlidingWindow-classified SDPA is guaranteed (by detect_causal_mask.cpp's
            // matchers) to already carry an explicit, statically-shaped mask input - none of them
            // are `is_causal`-only.
            NPUW_ASSERT(sdpa->get_input_size() > kMaskInputIdx);
            const ov::PartialShape mask_shape = sdpa->input(kMaskInputIdx).get_partial_shape();
            if (!sliding_mask_param) {
                NPUW_ASSERT(mask_shape.is_static());
                sliding_mask_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, mask_shape);
                set_mask_param_name(sliding_mask_param, "sliding_window_attention_mask");
            } else {
                NPUW_ASSERT(mask_shape == sliding_mask_param->get_partial_shape());
            }
            sdpa->input(kMaskInputIdx).replace_source_output(sliding_mask_param->output(0));
            ++num_cut;
        }
        if (sliding_mask_param) {
            model->add_parameters({sliding_mask_param});
            changed = true;
            LOG_INFO("[SWA] Externalized " << num_cut
                                           << " sliding SDPA mask input(s) as 'sliding_window_attention_mask' in '"
                                           << model->get_friendly_name() << "'.");
        }
    }

    // Step 0b: BEFORE touching any Parameter shape, snapshot the *current* (fully static, still
    // uniform-`m_kvcache_size`) value of EVERY self_attn layer's GQA/repeat_kv `expand` lowering's
    // Broadcast/Reshape target shape (`Concat(past,new) -> Unsqueeze -> Broadcast -> Reshape ->
    // SDPA`) - sliding *and* full-attention layers alike. This must happen strictly first, and
    // must cover ALL layers, for two reasons:
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
    //     unexpectedly showed the *sliding* size. So every layer's shape input touching this
    //     pattern must be decoupled from the (possibly shared) upstream computation, regardless
    //     of that layer's own sliding status.
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
    //
    // NOTE: the SLIDING SDPA's OWN mask input is no longer read from any in-graph subgraph for a
    // genuine SWA hybrid model - Step 0 above already redirected it to the
    // `sliding_window_attention_mask` model input instead, sized directly below in Step 1b. But
    // that does NOT mean the ORIGINAL mask-construction subgraph (e.g. a per-layer causal-mask
    // `Slice(mask, begin, end, ...)` trimming it to the current KV length) is irrelevant here: a
    // FULL-ATTENTION layer's own native mask Slice can be CSE-shared with a sliding layer's shape
    // computation exactly like the Broadcast/Reshape case above (same root cause: exporters
    // frequently reuse one ShapeOf/Gather chain derived from ONE particular layer's `past_key_values`
    // across MULTIPLE structurally-identical `layers.N.self_attn` blocks, sliding and full alike).
    // Once Step 1 shrinks a sliding layer's `past_key_values` Parameter, any full-attention layer's
    // Slice still wired to that SAME shared node silently inherits the sliding layer's new
    // (wrong-for-it) `new_kv_total`, instead of keeping the model's real `m_kvcache_size` - this is
    // the exact failure observed in practice: a global/full-attention SDPA's mask input ends up
    // `[.., window_size + input_size]` instead of `[.., kvcache_size]`. So this Slice's begin/end
    // bounds must be snapshotted and privatized the same way as Broadcast/Reshape below, for every
    // self_attn layer (sliding layers' own Slice is normally already dead/unreachable here since
    // Step 0 disconnected its SDPA consumer, so in practice only full-attention layers' Slice
    // nodes are found and rebuilt - but the logic stays symmetric/correct either way).
    struct ShapeInputSnapshot {
        std::shared_ptr<ov::Node> consumer;
        size_t input_index;
        size_t layer_idx;
        bool layer_is_sliding;
        ov::element::Type elem_type;
        std::vector<int64_t> original_values;
    };
    std::vector<ShapeInputSnapshot> shape_snapshots;
    static constexpr size_t kTargetShapeInputIdx = 1;  // Reshape/Broadcast target-shape input.
    auto snapshot_kvcache_dim_input = [&](const std::shared_ptr<ov::Node>& node, size_t input_idx) {
        size_t layer_idx = 0;
        if (!ov::npuw::util::try_parse_self_attn_layer_idx(node->get_friendly_name(), layer_idx)) {
            return;
        }
        if (layer_idx >= m_layer_is_sliding.size() || input_idx >= node->get_input_size()) {
            return;
        }
        auto folded = ov::util::get_constant_from_source(node->input_value(input_idx));
        if (!folded) {
            LOG_WARN("[SWA] Layer " << layer_idx << ": " << node->get_type_name() << " '"
                                    << node->get_friendly_name() << "' input(" << input_idx
                                    << ") is not constant-foldable, cannot verify/patch it.");
            return;
        }
        auto values = folded->cast_vector<int64_t>();
        const auto kv = static_cast<int64_t>(m_kvcache_size);
        const bool has_kvcache_dim =
            std::any_of(values.begin(), values.end(), [kv](int64_t v) { return v == kv || v == -kv; });
        if (!has_kvcache_dim) {
            // Not a KV-size-dependent shape/bound (e.g. unrelated input in the same self_attn
            // block) - nothing to guard here.
            return;
        }
        shape_snapshots.push_back(ShapeInputSnapshot{node,
                                                     input_idx,
                                                     layer_idx,
                                                     m_layer_is_sliding[layer_idx],
                                                     folded->get_element_type(),
                                                     std::move(values)});
    };
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v3::Broadcast>(node)) {
            snapshot_kvcache_dim_input(node, kTargetShapeInputIdx);
        } else if (ov::is_type<ov::op::v8::Slice>(node)) {
            // Slice(data, begin, end, step, [axes]): both `begin` (input 1) and `end` (input 2)
            // can independently carry a current-KV-length bound (e.g. a causal-mask trim's `end`,
            // or a sliding-window trim's `begin`) - check both.
            snapshot_kvcache_dim_input(node, 1);
            snapshot_kvcache_dim_input(node, 2);
        }
    }

    // Step 1: shrink past_key_values Parameter shapes for sliding-window layers.
    //
    // NOTE: we deliberately do NOT call `model->reshape()` here - see the rationale in Step 0's
    // comment above. Instead we set the new Parameter shape directly
    // (`Parameter::set_partial_shape`, which does not trigger any validation), and defer the
    // single `validate_nodes_and_infer_types()` call to the very end of this function, once Step 2
    // has also fixed up the Broadcast/Reshape target-shape inputs snapshotted in Step 0.
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

    // Step 1b (always performed, unconditionally, for BOTH prefill and generate variants): shrink
    // the `sliding_window_attention_mask` model input created by Step 0 above to the same
    // `new_kv_total` width as the past_key_values buffers above - the host-side runtime (see
    // Phase 5's mask-building code) is responsible for filling it with the correctly-shaped
    // window-relative mask content at inference time. Atomic with Step 1's past_key_values shrink
    // for the same reason (single deferred `validate_nodes_and_infer_types()` call at the end).
    //
    // This MUST run for prefill too (an earlier version of this pass skipped it for prefill,
    // reasoning that prefill's per-query-position staggered/banded mask content couldn't be
    // represented by a fixed-width shrink) - that reasoning conflated mask CONTENT (a host-fill
    // algorithm concern) with mask SHAPE (a hard graph-validation requirement). Step 1 above
    // already unconditionally shrinks every sliding layer's K/V total to `new_kv_total` -
    // including for prefill, e.g. chunked prefill where the past buffer is NOT empty - so the
    // shared `sliding_window_attention_mask` Parameter's last axis MUST match `new_kv_total` in
    // every variant, or `validate_nodes_and_infer_types()` fails with an attention-mask/K-V shape
    // mismatch at the consuming SDPA node (observed in practice on a real hybrid SWA model).
    for (const auto& input : model->inputs()) {
        if (input.get_any_name() != "sliding_window_attention_mask") {
            continue;
        }
        const auto& pshape = input.get_partial_shape();
        if (pshape.rank().is_dynamic() || pshape.size() == 0 || !pshape[pshape.size() - 1].is_static()) {
            LOG_WARN("[SWA] 'sliding_window_attention_mask' has a dynamic last axis, skipping shrink. Was "
                     "ReshapeToStatic applied before PatchSlidingWindowKVCache?");
            break;
        }
        const size_t last_axis = pshape.size() - 1;
        const int64_t old_width = pshape[last_axis].get_length();
        if (new_kv_total == old_width) {
            break;  // already the right width (e.g. window_size >= kvcache_size for this variant).
        }
        auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(input.get_node_shared_ptr());
        if (!param) {
            LOG_WARN("[SWA] 'sliding_window_attention_mask' is not backed by a Parameter node, skipping.");
            break;
        }
        ov::PartialShape new_shape = pshape;
        new_shape[last_axis] = new_kv_total;
        param->set_partial_shape(new_shape);
        changed = true;
        LOG_INFO("[SWA] 'sliding_window_attention_mask' last axis " << old_width << " -> " << new_kv_total
                                                                    << " in '" << model->get_friendly_name()
                                                                    << "'.");
        break;
    }

    // Step 2: apply the fix for every Broadcast/Reshape target shape snapshotted in Step 0. For
    // each, take its ORIGINAL (pre-patch) value and rebuild it as a brand-new, PRIVATE Constant
    // via `replace_source_output()` - never mutating the original (possibly shared) upstream node
    // in place:
    //   - sliding-window layers: entries equal to `m_kvcache_size` are corrected to `new_kv_total`.
    //   - full-attention layers: entries are kept at their original value - but STILL rebuilt as a
    //     private Constant, purely to sever any (possibly shared) live upstream dependency that a
    //     sibling sliding layer might later invalidate (see Step 0's comment for why this matters
    //     even when the value itself doesn't change).
    {
        size_t patched_count = 0;
        size_t privatized_count = 0;
        for (const auto& snapshot : shape_snapshots) {
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
            auto new_const =
                std::make_shared<ov::op::v0::Constant>(snapshot.elem_type, ov::Shape{values.size()}, values);
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
                                         << " Broadcast/Reshape target-shape input(s) (" << patched_count
                                         << " value-corrected, " << (privatized_count - patched_count)
                                         << " guarded-only) feeding self_attn SDPA node(s) in '"
                                         << model->get_friendly_name() << "'.");
        }
    }

    if (changed) {
        model->validate_nodes_and_infer_types();
    }
    return changed;
}

}  // namespace ov::npuw
