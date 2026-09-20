// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "json_object.h"
#include "primitive_type_base.h"
#include "qsa_inst.h"
#include "to_string_utils.h"

namespace cldnn {

GPU_DEFINE_PRIMITIVE_TYPE_ID(qsa)

layout qsa_inst::calc_output_layout(const qsa_node& node, const kernel_impl_params& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param)[0];
}

template <typename ShapeType>
std::vector<layout> qsa_inst::calc_output_layouts(const qsa_node& /*node*/, const kernel_impl_params& impl_param) {
    const auto& desc = impl_param.typed_desc<qsa>();

    if (desc->state_only) {
        // QSAIndexerCache lowering (see ops/qsa_indexer_cache.cpp): a pure state
        // DECLARATION dispatches zero kernel stages, so its two declared outputs are
        // simply a pass-through of its two INDEXER_RAW_K_CACHE/INDEXER_SUMMARY_CACHE
        // inputs -- no growth math (that would need this call's new token count, which a
        // state declaration never receives) and no mutation.
        const auto& raw_layout = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_RAW_K_CACHE);
        const auto& summary_layout = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_SUMMARY_CACHE);
        OPENVINO_ASSERT(raw_layout.get<ShapeType>().rank().is_static() && raw_layout.get<ShapeType>().size() == 3 &&
                            summary_layout.get<ShapeType>().rank().is_static() && summary_layout.get<ShapeType>().size() == 3,
                        "[GPU] qsa state_only expects rank-3 [B, L, Di] raw-K/summary cache inputs");
        return {raw_layout, summary_layout};
    }

    // Shared by prepare_only+dense_mode (QSAIndexerCache's now-retired growth path, kept
    // for potential direct/primitive-level use) and !prepare_only+dense_mode (QSASDPA's
    // own dense dispatch, which now owns reserve/grow/Q1-write 
    // and ops/qsa_sdpa.cpp::CreateQSASDPAOp): the OLD (small) raw-K/summary history is fed
    // as INDEXER_RAW_K_CACHE/INDEXER_SUMMARY_CACHE; the declared output is the GROWN
    // buffer (old_length + this call's new entries), computed purely from already-resolved
    // input shapes -- the same "read impl_param.get_input_layout(...) at shape-inference
    // time" pattern kv_cache_inst::compute_trim_length uses for its own dynamic shape
    // inference, just without needing to read any tensor's VALUES (raw-K grows by exactly
    // this call's new token count -- MIXED's own row count -- and summary grows to
    // floor(new_raw_length / r), which only depends on lengths, not content). execute()'s
    // copy_from() (see impls/cm/qsa/qsa.cpp) then populates the new buffer's valid prefix
    // from the old one before Q1 writes/reads it.
    auto compute_dense_growth_layouts = [&]() -> std::vector<layout> {
        const auto old_raw_shape = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_RAW_K_CACHE).get<ShapeType>();
        const auto old_summary_shape = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_SUMMARY_CACHE).get<ShapeType>();
        OPENVINO_ASSERT(old_raw_shape.rank().is_static() && old_raw_shape.size() == 3 && old_summary_shape.rank().is_static() &&
                            old_summary_shape.size() == 3,
                        "[GPU] qsa dense_mode expects rank-3 [B, L, Di] raw-K/summary cache inputs");
        const auto raw_dt = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_RAW_K_CACHE).data_type;
        const auto summary_dt = impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_SUMMARY_CACHE).data_type;

        const auto mixed_shape = impl_param.get_input_layout(qsa::QSAInputIdx::MIXED).get<ShapeType>();
        // Graph-build-time shape propagation (compile_model()'s initial static pass,
        // before any concrete per-call runtime shape is known) calls this same
        // function with MIXED's/old_raw_shape's dynamic dimensions still
        // unresolved -- gracefully fall back to a fully dynamic declared output
        // shape in that case; the EXACT length is only computable once real,
        // per-call values are available (the ordinary runtime shape-inference pass
        // that precedes every execute()).
        if (!mixed_shape.rank().is_static() || mixed_shape.size() < 1 || !mixed_shape[0].is_static() || !old_raw_shape[1].is_static()) {
            ShapeType raw_dyn{old_raw_shape[0], ov::Dimension::dynamic(), old_raw_shape[2]};
            ShapeType summary_dyn{old_summary_shape[0], ov::Dimension::dynamic(), old_summary_shape[2]};
            return {layout{raw_dyn, raw_dt, cldnn::format::get_default_format(3)}, layout{summary_dyn, summary_dt, cldnn::format::get_default_format(3)}};
        }

        const int64_t old_raw_length = old_raw_shape[1].get_length();
        const int64_t new_token_count = mixed_shape[0].get_length();
        const int64_t new_raw_length = old_raw_length + new_token_count;
        const int64_t r = static_cast<int64_t>(desc->compress_ratio);
        OPENVINO_ASSERT(r > 0, "[GPU] qsa dense_mode requires compress_ratio > 0");
        const int64_t new_summary_length = new_raw_length / r;  // floor: total completed blocks so far

        ShapeType raw_out{old_raw_shape[0], ov::Dimension(new_raw_length), old_raw_shape[2]};
        ShapeType summary_out{old_summary_shape[0], ov::Dimension(new_summary_length), old_summary_shape[2]};
        return {layout{raw_out, raw_dt, cldnn::format::get_default_format(3)}, layout{summary_out, summary_dt, cldnn::format::get_default_format(3)}};
    };

    if (desc->prepare_only) {
        // Q1 mutates the INDEXER_RAW_K_CACHE / INDEXER_SUMMARY_CACHE inputs in place, so
        // the primitive's two declared outputs are simply those same (post-mutation)
        // buffers -- there is no separate "attention output" to compute when Q2/Q3 never
        // run. Declaring them as real outputs (rather than leaving the mutation implicit)
        // gives downstream consumers a genuine graph dependency edge to schedule on; see
        // execute()'s set_output_memory() calls.
        if (desc->dense_mode) {
            return compute_dense_growth_layouts();
        }

        // The declared shape is the dense 3D [num_blocks, pa_block_size, Di] form (the
        // paged 4D [num_blocks, 1, pa_block_size, Di] input's head axis, always 1, is
        // squeezed out) rather than a literal copy of the paged input layout: aliasing
        // (execute()'s set_output_memory()) only swaps a memory pointer, so this is a free
        // choice of *declared* rank with no effect on the underlying bytes.
        auto squeeze_head_axis = [](const layout& l) {
            auto shape = l.get<ShapeType>();
            OPENVINO_ASSERT(shape.rank().is_static() && shape.size() == 4,
                            "[GPU] qsa prepare_only expects a rank-4 [num_blocks, 1, pa_block_size, Di] cache input, got ", l.to_short_string());
            return layout{ShapeType{shape[0], shape[2], shape[3]}, l.data_type, cldnn::format::get_default_format(3)};
        };
        return {squeeze_head_axis(impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_RAW_K_CACHE)),
                squeeze_head_axis(impl_param.get_input_layout(qsa::QSAInputIdx::INDEXER_SUMMARY_CACHE))};
    }

    const auto query_layout = impl_param.get_input_layout(qsa::QSAInputIdx::QUERY);

    auto out_shape = ShapeType{query_layout.get<ShapeType>()};
    // A logically rank-2 [tokens, Hq*Dh] input may be reported with extra trailing size-1
    // dims: ProgramBuilder's legacy static-shape flow (ProgramBuilder::use_new_shape_infer()
    // == false, the default whenever every input in the compiled ov::Model happens to have
    // a fully static shape) pads any rank < 4 input up to 4 dimensions with trailing 1s
    // (see plugin/ops/parameter.cpp::CreateParameterOp) before this function ever sees it.
    // Both representations are accepted since index 0/1 (tokens / Hq*Dh) carry the correct
    // values either way; only a genuinely different logical rank (a non-singleton dim
    // beyond index 1) is rejected.
    OPENVINO_ASSERT(out_shape.rank().is_static() && out_shape.size() >= 2 &&
                       std::all_of(out_shape.begin() + 2, out_shape.end(), [](const ov::Dimension& dim) { return dim.compatible(1); }),
                    "[GPU] qsa expects a rank-2 [tokens, Hq*Dh] query input, got ",
                    query_layout.to_short_string(),
                    " (num inputs=",
                    impl_param.input_layouts.size(),
                    ")");
    out_shape[1] = ov::Dimension(static_cast<int64_t>(desc->heads_num * desc->v_head_size));

    const auto out_dt =
        desc->output_data_types.empty() || !desc->output_data_types[0].has_value() ? query_layout.data_type : desc->output_data_types[0].value();

    const layout attn_layout{out_shape, out_dt, query_layout.format};

    if (desc->dense_mode && desc->grow_dense_state) {
        // QSASDPA's own dense dispatch owns growth of the persistent indexer raw-K/
        // summary state only when explicitly opted in via
        // grow_dense_state (set by ops/qsa_sdpa.cpp when its indexer cache inputs come
        // from a real QSAIndexerCache node). Outputs 1/2, alongside the attention result
        // at output 0, are the grown buffers that CreateQSASDPAOp Assigns back into
        // QSAIndexerCache's Variables. When grow_dense_state is false (the "advanced
        // caller manages all cache state externally" mode), dense_mode keeps its
        // original single-output, in-place-mutate contract unchanged.
        auto growth = compute_dense_growth_layouts();
        return {attn_layout, growth[0], growth[1]};
    }
    return {attn_layout};
}


template std::vector<layout> qsa_inst::calc_output_layouts<ov::PartialShape>(const qsa_node& node, const kernel_impl_params& impl_param);

std::string qsa_inst::to_string(const qsa_node& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    info.add("heads_num", desc->heads_num);
    info.add("kv_heads_num", desc->kv_heads_num);
    info.add("k_head_size", desc->k_head_size);
    info.add("v_head_size", desc->v_head_size);
    info.add("hidden_size", desc->hidden_size);
    info.add("indexer_heads", desc->indexer_heads);
    info.add("indexer_kv_heads", desc->indexer_kv_heads);
    info.add("indexer_head_dim", desc->indexer_head_dim);
    info.add("indexer_rotary_dim", desc->indexer_rotary_dim);
    info.add("budget", desc->budget);
    info.add("compress_ratio", desc->compress_ratio);
    info.add("qsa_block_tokens", desc->qsa_block_tokens);
    info.add("pa_block_size", desc->pa_block_size);
    info.add("block_topk", desc->block_topk);
    info.add("scale", desc->scale);
    info.add("has_output_gate", desc->has_output_gate);
    info.add("is_causal", desc->is_causal);

    node_info->add("qsa_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

qsa_inst::typed_primitive_inst(network& network, const qsa_node& node) : parent(network, node) {
    std::string reason;
    OPENVINO_ASSERT(ov::intel_gpu::op::QSA::validate_config(node.get_primitive()->get_config(), reason), "[GPU] Invalid qsa primitive: ", reason);
}

}  // namespace cldnn
