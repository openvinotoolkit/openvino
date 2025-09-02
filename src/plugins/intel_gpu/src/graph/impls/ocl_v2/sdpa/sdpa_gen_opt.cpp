// Copyright (C) 2025 Intel Corporation
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
        GPU_DEBUG_TRACE_DETAIL << "k_head_size = " << k_head_size << ", v_head_size = " << v_head_size << "\n";

        size_t data_inputs_num = get_data_inputs_num(*desc);
        size_t scale_idx = 4;
        if ((data_inputs_num > scale_idx) && (!desc->scale_val.has_value())) {
            jit.make("HAS_SCALE_INPUT", 1);
        } else if (desc->scale_val.has_value()) {
            jit.make("STATIC_SCALE_VALUE", desc->scale_val.value());
            jit.make("STATIC_SCALE_VALUE_INV", 1.0f / desc->scale_val.value());
        } else {
            jit.make("STATIC_SCALE_VALUE_INV", std::sqrt(static_cast<float>(k_head_size)));
            jit.make("STATIC_SCALE_VALUE", 1.0f / std::sqrt(static_cast<float>(k_head_size)));
        }
        // jit.make("NUM_KV_HEADS", -1);
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

    const size_t attn_mask_idx = 3;
    const size_t scale_idx = 4;
    for (uint32_t i = 0; i < data_inputs_num; i++) {
        if (i == attn_mask_idx && desc->attn_mask_val.has_value())
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

Arguments SDPAOptGeneratorSingleToken::get_arguments_desc(const kernel_impl_params& params) const {
    return get_arguments_desc_impl(params, SDPAStage::SINGLE_TOKEN);
}

JitConstants SDPAOptGeneratorSingleToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::SINGLE_TOKEN);
    jit.make("SDPA_STAGE_0", 1);
    jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", 1);
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
            const size_t num_of_partitions = get_partitions_num(params, SDPAStage::SINGLE_TOKEN);
            const auto head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);
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
            const size_t head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);
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
    return get_arguments_desc_impl(params, SDPAStage::FINALIZATION);
}

JitConstants SDPAOptGeneratorFinalization::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::FINALIZATION);
    jit.make("SDPA_STAGE_1", 1);
    jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());
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
            const size_t num_of_partitions = get_partitions_num(params, SDPAStage::FINALIZATION);
            const size_t head_size = get_head_size(params.get_input_layout(2), extended_input_v_transpose_order);

            GPU_DEBUG_TRACE_DETAIL << "batch_size = " << batch_size << ", target_seq_len = " << target_seq_len << ", heads_num = " << heads_num << "\n";
            GPU_DEBUG_TRACE_DETAIL << "head_size = " << head_size << ", num_of_partitions = " << num_of_partitions << "\n";

            wgs.global = {batch_size * heads_num, target_seq_len, head_size};
            wgs.local = {1, 1, head_size};
            num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);
            scalars.push_back(num_of_partitions_scalar);
        }
    }};
}

}  // namespace ov::intel_gpu::ocl
