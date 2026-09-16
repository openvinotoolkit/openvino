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

    if (desc->prepare_only) {
        // QSAIndexerCache lowering (see QSA_SDPA_DESIGN.md section 4.3): Q1 mutates the
        // INDEXER_RAW_K_CACHE / INDEXER_SUMMARY_CACHE inputs in place, so the primitive's
        // two declared outputs are simply those same (post-mutation) buffers -- there is
        // no separate "attention output" to compute when Q2/Q3 never run. Declaring them
        // as real outputs (rather than leaving the mutation implicit) gives downstream
        // consumers (e.g. an Assign writing back to persistent Variable state) a genuine
        // graph dependency edge to schedule on; see execute()'s set_output_memory() calls.
        //
        // The declared shape is the dense 3D [num_blocks, pa_block_size, Di] form (the
        // paged 4D [num_blocks, 1, pa_block_size, Di] input's head axis, always 1, is
        // squeezed out) rather than a literal copy of the paged input layout: aliasing
        // (execute()'s set_output_memory()) only swaps a memory pointer, so this is a free
        // choice of *declared* rank with no effect on the underlying bytes, and 3D is what
        // QSAIndexerCache's own Variable-backed state (and its OV-level declared output
        // shape) is declared as -- avoiding a redundant reshape pair in its lowering.
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

    return {layout{out_shape, out_dt, query_layout.format}};
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
