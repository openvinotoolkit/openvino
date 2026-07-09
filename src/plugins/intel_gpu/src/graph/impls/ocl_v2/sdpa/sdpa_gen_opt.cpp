// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_gen_opt.hpp"

#include "../utils/jitter.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_opt.hpp"

namespace ov::intel_gpu::ocl {

static bool unaligned_head_size(const size_t k_head_size, const size_t v_head_size, const size_t subgroup_size) {
    return (k_head_size % subgroup_size != 0) || (v_head_size % subgroup_size != 0);
}

JitConstants SDPAOptGeneratorBase::get_jit_constants_base(const kernel_impl_params& params, size_t stage, bool add_tensor_definitions) const {
    auto jit = SDPABase::get_jit_constants(params);
    const auto& info = params.get_device_info();
    const bool is_paged_attention = params.is_type<paged_attention>() ? true : false;

    if (add_tensor_definitions) {
        jit.add(make_tensors_jit_constants(params));
    }

    constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
    jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", softmax_accumulator_type));
    constexpr size_t subgroup_size = 16;
    jit.make("SUBGROUP_SIZE", subgroup_size);

    auto [broadcast_axis, group_size] = get_gqa_params(params);
    int64_t v_head_size = -1, k_head_size = -1;

    if (is_paged_attention) {
        auto desc = params.typed_desc<paged_attention>();
        v_head_size = static_cast<int64_t>(desc->v_head_size);
        k_head_size = static_cast<int64_t>(desc->k_head_size);
        jit.make("IS_PAGED_ATTENTION", 1);
        jit.make("NUM_KV_HEADS", desc->kv_heads_num);
        if (desc->has_alibi) {
            jit.make("HAS_ALIBI", 1);
        }

        if (params.output_layouts.size() > 1) {
            jit.make("PAGED_ATTENTION_SCORES_OUTPUT", 1);
        }

        if (desc->has_score_aggregation) {
            jit.make("HAS_SCORE_AGGREGATION", 1);
        }
    } else {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        const auto& k_layout = params.get_input_layout(1);
        const auto& v_layout = params.get_input_layout(2);

        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        k_head_size = get_head_size(k_layout, extended_input_k_transpose_order);
        v_head_size = get_head_size(v_layout, extended_input_v_transpose_order);

        // 4-bit KV-cache: K/V layouts have head_size/2 due to u4→i8 packing.
        // Override with logical head size from query (which is not packed).
        {
            if (desc->is_kv_compressed && SDPABase::is_int4_kv_cache(params)) {
                auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
                auto q_head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
                k_head_size = q_head_size;
                v_head_size = q_head_size;
            }
        }

        GPU_DEBUG_TRACE_DETAIL << "k_head_size = " << k_head_size << ", v_head_size = " << v_head_size << "\n";

        size_t data_inputs_num = get_data_inputs_num(*desc);
        size_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
        const bool has_attn_mask_input = sdpa_has_runtime_attn_mask_input(params);
        if (desc->attn_mask_val.has_value()) {
            jit.make("STATIC_SCALAR_ATTN_MASK_VALUE", desc->attn_mask_val.value());
            jit.make("HAS_ATTN_MASK_INPUT", 0);
        } else {
            jit.make("HAS_ATTN_MASK_INPUT", has_attn_mask_input ? 1 : 0);
            if (has_attn_mask_input) {
                const auto& attn_mask_layout = params.get_input_layout(attn_mask_idx);
                // Enable clamping only if attn_mask dtype differs from softmax_accumulator_type(f32)
                if (attn_mask_layout.data_type != data_types::f32) {
                    jit.make("CLAMP_ATTN_MASK_INPUT", 1);
                }
            }
        }
        size_t scale_idx = ScaledDotProductAttentionInputIdx::SCALE;
        if ((data_inputs_num > scale_idx) && (!desc->scale_val.has_value())) {
            jit.make("HAS_SCALE_INPUT", 1);
        } else if (desc->scale_val.has_value()) {
            jit.make("STATIC_SCALE_VALUE", desc->scale_val.value());
            jit.make("STATIC_SCALE_VALUE_INV", 1.0f / desc->scale_val.value());
        } else {
            jit.make("STATIC_SCALE_VALUE_INV", std::sqrt(static_cast<float>(k_head_size)));
            jit.make("STATIC_SCALE_VALUE", 1.0f / std::sqrt(static_cast<float>(k_head_size)));
        }
        if (info.supports_immad && broadcast_axis == -1 && k_head_size >= 128) {
            jit.make("LOAD_KEY_LEFTOVERS_IN_CALC_LOOP", 1);
        }
    }

    if (unaligned_head_size(k_head_size, v_head_size, subgroup_size)) {
        jit.make("K_HEAD_SIZE_LEFTOVER", k_head_size % subgroup_size);
        jit.make("V_HEAD_SIZE_LEFTOVER", v_head_size % subgroup_size);
    }
    jit.make("SEQ_LEN_PARTITION_SIZE", get_seq_len_partition_size(info, v_head_size, stage));
    jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(info, v_head_size, stage));

    bool could_use_flashattn_v2 = params.get_program().get_config().get_could_use_flashattn_v2();
    if (could_use_flashattn_v2) {
        jit.make("IS_FLASHATTEN_V2", 1);
    }
    return jit;
}

Arguments SDPAOptGeneratorBase::get_arguments_desc_impl(const kernel_impl_params& params, size_t stage) const {
    Arguments args;
    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }

    auto desc = params.typed_desc<scaled_dot_product_attention>();
    size_t data_inputs_num = stage == SDPAStage::FINALIZATION ? 0 : get_data_inputs_num(*desc);
    auto has_zp_input_buffers = desc->get_compression_zp_inputs_num() > 0;

    const size_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
    const size_t scale_idx = ScaledDotProductAttentionInputIdx::SCALE;
    const bool has_attn_mask_input = sdpa_has_runtime_attn_mask_input(params);
    for (uint32_t i = 0; i < data_inputs_num; i++) {
        if (i == attn_mask_idx && !has_attn_mask_input)
            continue;
        if (i == scale_idx && desc->scale_val.has_value())
            continue;
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    auto beam_table_idx = data_inputs_num;
    if (desc->is_kv_compressed && stage != SDPAStage::FINALIZATION) {
        auto key_cache_compression_scale_idx = static_cast<uint32_t>(data_inputs_num);
        auto value_cache_compression_scale_idx = static_cast<uint32_t>(data_inputs_num + 1);

        args.push_back({ArgumentDescriptor::Types::INPUT, key_cache_compression_scale_idx});
        args.push_back({ArgumentDescriptor::Types::INPUT, value_cache_compression_scale_idx});

        if (has_zp_input_buffers) {
            args.push_back({ArgumentDescriptor::Types::INPUT, key_cache_compression_scale_idx + 2});
            args.push_back({ArgumentDescriptor::Types::INPUT, value_cache_compression_scale_idx + 2});
            beam_table_idx += 2;
        }

        beam_table_idx += 2;
    }

    if (m_indirect && stage != SDPAStage::FINALIZATION) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(beam_table_idx)});
    }

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

    if (stage == SDPAStage::FINALIZATION) {
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
    }

    return args;
}

// ============================================================================================
// split-KV decode helpers. The split-KV decode path reuses the single-token + finalization opt
// generators below (same sdpa_opt.cl template); when desc->split_kv is set they emit a few deltas
// via these helpers: the SPLIT_KV jit (+ KEY_NEW/VALUE_NEW layout macros, SOURCE_SEQ_LEN_CACHE), a
// different argument list (no scale input; trailing K_new/V_new/kv_len), and one extra partition
// for the new chunk. split_kv is part of the op hash, so split / non-split kernels never collide.
// ============================================================================================
namespace {

// split_kv inputs end with [.., K_new, V_new, kv_len], so K_new/V_new are the 3rd/2nd from last
// (the last input is the i32 kv_len control tensor).
size_t split_kv_k_new_idx(const scaled_dot_product_attention& desc) {
    return desc.input_size() - 3;
}
size_t split_kv_v_new_idx(const scaled_dot_product_attention& desc) {
    return desc.input_size() - 2;
}

bool is_default_order(const std::vector<int64_t>& order) {
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

// Emit the split-KV jit deltas on top of the single-token base jit: SPLIT_KV, the KEY_NEW/VALUE_NEW
// layout macros (so the kernel indexes the new chunk exactly like INPUT1/INPUT2), SOURCE_SEQ_LEN_CACHE
// (the static cache boundary), and the redefined SOURCE_SEQ_LEN = cache + new.
void add_split_kv_jit_constants(JitConstants& jit, const kernel_impl_params& params, const scaled_dot_product_attention& desc) {
    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const uint32_t k_new_idx = static_cast<uint32_t>(split_kv_k_new_idx(desc));
    const uint32_t v_new_idx = static_cast<uint32_t>(split_kv_v_new_idx(desc));

    jit.add(make_layout_jit_constants("KEY_NEW", params.input_layouts[k_new_idx], in_offsets_map.at(k_new_idx)));
    jit.add(make_layout_jit_constants("VALUE_NEW", params.input_layouts[v_new_idx], in_offsets_map.at(v_new_idx)));

    const auto extended_k_order = extend_order_in_num_heads_dim(desc.input_k_transpose_order);
    const auto extended_v_order = extend_order_in_num_heads_dim(desc.input_v_transpose_order);
    if (!extended_k_order.empty() && !is_default_order(extended_k_order)) {
        jit.make("KEY_NEW_DIMS_ORDER", get_dims_order(extended_k_order));
    }
    if (!extended_v_order.empty() && !is_default_order(extended_v_order)) {
        jit.make("VALUE_NEW_DIMS_ORDER", get_dims_order(extended_v_order));
    }

    // Seq axis is Y in transposed [B,H,S,D] space. SOURCE_SEQ_LEN_CACHE = K_cache seq; redefine
    // SOURCE_SEQ_LEN = cache + new (string exprs valid for dynamic shapes, though this path is static).
    const auto updated_params = SDPABase::static_canonicalize_shapes(params);
    LayoutJitter k_new_jitter(updated_params.input_layouts[k_new_idx], in_offsets_map.at(k_new_idx));
    const auto new_seq = k_new_jitter.dim(get_transposed_channel(ChannelName::Y, extended_k_order));
    LayoutJitter k_cache_jitter(updated_params.input_layouts[1], in_offsets_map.at(1));
    const auto cache_seq = k_cache_jitter.dim(get_transposed_channel(ChannelName::Y, extended_k_order));

    jit.make("SPLIT_KV", 1);
    jit.make("SOURCE_SEQ_LEN_CACHE", cache_seq);
    jit.remove("SOURCE_SEQ_LEN");
    jit.make("SOURCE_SEQ_LEN", "(" + cache_seq + " + " + new_seq + ")");
}

// Build the split-KV argument list: [Q, K_cache, V_cache, (mask), K_new, V_new, kv_len, output,
// exp_sums, max_logits, tmp_out]. The new chunk + kv_len are inserted between the base inputs and
// the output; the finalization stage takes only the buffers + num_of_partitions scalar. There is
// no scale input (scale is baked via STATIC_SCALE_VALUE) and no compression / beam table.
Arguments split_kv_arguments(const kernel_impl_params& params, bool is_finalization) {
    Arguments args;
    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }
    auto desc = params.typed_desc<scaled_dot_product_attention>();

    if (!is_finalization) {
        const size_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
        const bool has_attn_mask_input = sdpa_has_runtime_attn_mask_input(params);
        // Base inputs Q, K_cache, V_cache, [mask] (split_kv has no scale input).
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::VALUE});
        if (has_attn_mask_input) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(attn_mask_idx)});
        }
        // Trailing split-KV inputs in kernel-arg order: K_new, V_new, kv_len.
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(split_kv_k_new_idx(*desc))});
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(split_kv_v_new_idx(*desc))});
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(desc->input_size() - 1)});
    }

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // exp_sums
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // max_logits
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // tmp_out

    if (is_finalization) {
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
    }
    return args;
}

// Number of decode partitions for split-KV: sdpa_opt partitions the static cache seq into
// ceil(S_cache / SEQ_LEN_PARTITION_SIZE), and the new chunk gets ONE additional partition (it is
// short -- S_new tokens, typically 1). Cache partitions beyond the runtime valid length early-exit
// in the kernel, so dispatching all static cache partitions is correct (and dynamic-value-free).
size_t split_kv_partitions_num(const kernel_impl_params& params) {
    return get_partitions_num(params, SDPAStage::SINGLE_TOKEN) + 1;
}

}  // namespace

bool split_kv_opt_supported(const kernel_impl_params& params) {
    if (params.is_dynamic()) {
        return false;
    }
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (!desc->split_kv) {
        return false;
    }
    // Default contiguous [B,H,S,D] K/V only (matches the fusion's layout restriction); the SPLIT_KV
    // split-load assumes the head dim is contiguous. The new chunk gets its own partition
    // (index ceil(S_cache / partition_size)), so S_cache need NOT be partition-aligned: the last
    // cache partition is just a short unaligned tail (sdpa_opt already handles that), and the new
    // partition remaps its mask column to the true logical position S_cache + j.
    const auto extended_q_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
    const auto extended_k_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
    const auto extended_v_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
    return is_default_order(extended_q_order) && is_default_order(extended_k_order) && is_default_order(extended_v_order);
}

Arguments SDPAOptGeneratorSingleToken::get_arguments_desc(const kernel_impl_params& params) const {
    if (params.typed_desc<scaled_dot_product_attention>()->split_kv) {
        return split_kv_arguments(params, /*is_finalization=*/false);
    }
    return get_arguments_desc_impl(params, SDPAStage::SINGLE_TOKEN);
}

JitConstants SDPAOptGeneratorSingleToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::SINGLE_TOKEN);
    jit.make("SDPA_STAGE_0", 1);
    jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", 1);
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (desc->split_kv) {
        add_split_kv_jit_constants(jit, params, *desc);
    }
    return jit;
}

DispatchDataFunc SDPAOptGeneratorSingleToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto params = SDPABase::requires_shape_canonicalization(impl_param) ? SDPABase::static_canonicalize_shapes(impl_param) : impl_param;
        if (!params.is_dynamic()) {
            auto desc = params.typed_desc<scaled_dot_product_attention>();
            auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
            auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
            auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

            const size_t batch_size = get_batch_size(params.get_output_layout(0), extended_output_transpose_order);
            const size_t target_seq_len = get_seq_length(params.get_input_layout(0), extended_input_q_transpose_order);
            const size_t heads_num = get_num_heads(params.get_output_layout(0), extended_output_transpose_order);
            // split-KV adds one partition for the new chunk; see split_kv_partitions_num.
            const size_t num_of_partitions =
                desc->split_kv ? split_kv_partitions_num(params) : get_partitions_num(params, SDPAStage::SINGLE_TOKEN);
            auto head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);

            // 4-bit KV-cache: V layout has head_size/2 due to u4→i8 packing.
            // Use logical head size from query for work-group dispatch.
            if (desc->is_kv_compressed && SDPABase::is_int4_kv_cache(params)) {
                head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
            }

            const size_t sg_num_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::SINGLE_TOKEN);
            GPU_DEBUG_TRACE_DETAIL << "batch_size = " << batch_size << ", target_seq_len = " << target_seq_len << ", heads_num = " << heads_num << "\n";
            GPU_DEBUG_TRACE_DETAIL << "head_size = " << head_size << ", num_of_partitions = " << num_of_partitions << "\n";
            GPU_DEBUG_TRACE_DETAIL << "sg_num_scale = " << sg_num_scale << "\n";

            wgs.global = {batch_size * heads_num, target_seq_len, head_size * num_of_partitions * sg_num_scale};
            wgs.local = {1, 1, head_size * sg_num_scale};
        }
    }};
}

Arguments SDPAOptGeneratorMultiToken::get_arguments_desc(const kernel_impl_params& params) const {
    return get_arguments_desc_impl(params, SDPAStage::MULTI_TOKENS);
}

JitConstants SDPAOptGeneratorMultiToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::MULTI_TOKENS);
    jit.make("SDPA_STAGE_0", 1);
    jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());

    return jit;
}

DispatchDataFunc SDPAOptGeneratorMultiToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto params = SDPABase::requires_shape_canonicalization(impl_param) ? SDPABase::static_canonicalize_shapes(impl_param) : impl_param;
        if (!params.is_dynamic()) {
            auto desc = params.typed_desc<scaled_dot_product_attention>();

            auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
            auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
            auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

            const size_t batch_size = get_batch_size(params.get_output_layout(0), extended_output_transpose_order);
            const size_t target_seq_len = get_seq_length(params.get_input_layout(0), extended_input_q_transpose_order);
            const size_t heads_num = get_num_heads(params.get_output_layout(0), extended_output_transpose_order);
            const size_t target_seq_len_block_size = get_target_seq_len_block_size();
            auto head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);

            // 4-bit KV-cache: V layout has head_size/2 due to u4→i8 packing.
            // Use logical head size from query for work-group dispatch.
            if (desc->is_kv_compressed && SDPABase::is_int4_kv_cache(params)) {
                head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
            }

            const size_t sg_num_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::MULTI_TOKENS);

            GPU_DEBUG_TRACE_DETAIL << "batch_size = " << batch_size << ", target_seq_len = " << target_seq_len << ", heads_num = " << heads_num << "\n";
            GPU_DEBUG_TRACE_DETAIL << "head_size = " << head_size << ", sg_num_scale = " << sg_num_scale << "\n";

            const size_t subgroup_size = 16;
            wgs.global = {batch_size * heads_num, ceil_div(target_seq_len, target_seq_len_block_size), align_to(head_size * sg_num_scale, subgroup_size)};
            wgs.local = {1, 1, align_to(head_size * sg_num_scale, subgroup_size)};
        }
    }};
}

Arguments SDPAOptGeneratorFinalization::get_arguments_desc(const kernel_impl_params& params) const {
    if (params.typed_desc<scaled_dot_product_attention>()->split_kv) {
        return split_kv_arguments(params, /*is_finalization=*/true);
    }
    return get_arguments_desc_impl(params, SDPAStage::FINALIZATION);
}

JitConstants SDPAOptGeneratorFinalization::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::FINALIZATION);
    jit.make("SDPA_STAGE_1", 1);
    jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (desc->split_kv) {
        add_split_kv_jit_constants(jit, params, *desc);
    }
    return jit;
}

DispatchDataFunc SDPAOptGeneratorFinalization::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();

        ScalarDescriptor num_of_partitions_scalar;
        num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;
        auto params = SDPABase::requires_shape_canonicalization(impl_param) ? SDPABase::static_canonicalize_shapes(impl_param) : impl_param;
        if (!params.is_dynamic()) {
            auto desc = params.typed_desc<scaled_dot_product_attention>();

            auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
            auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
            auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);
            const size_t batch_size = get_batch_size(params.get_output_layout(0), extended_output_transpose_order);
            const size_t target_seq_len = get_seq_length(params.get_input_layout(0), extended_input_q_transpose_order);
            const size_t heads_num = get_num_heads(params.get_output_layout(0), extended_output_transpose_order);
            // split-KV adds one partition for the new chunk; see split_kv_partitions_num.
            const size_t num_of_partitions =
                desc->split_kv ? split_kv_partitions_num(params) : get_partitions_num(params, SDPAStage::FINALIZATION);
            auto head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);

            // 4-bit KV-cache: V layout has head_size/2 due to u4→i8 packing.
            // Use logical head size from query for finalization dispatch.
            if (desc->is_kv_compressed && SDPABase::is_int4_kv_cache(params)) {
                head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
            }

            GPU_DEBUG_TRACE_DETAIL << "batch_size = " << batch_size << ", target_seq_len = " << target_seq_len << ", heads_num = " << heads_num << "\n";
            GPU_DEBUG_TRACE_DETAIL << "head_size = " << head_size << ", num_of_partitions = " << num_of_partitions << "\n";

            wgs.global = {batch_size * heads_num, target_seq_len, static_cast<size_t>(head_size)};
            wgs.local = {1, 1, static_cast<size_t>(head_size)};
            num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);
            scalars.push_back(num_of_partitions_scalar);
        }
    }};
}

}  // namespace ov::intel_gpu::ocl
