// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sdpa_opt.hpp"
#include "sdpa_base.hpp"
#include "common_utils/jitter.hpp"
#include "utils/kernel_generator.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

constexpr size_t subgroup_size = 16;

inline size_t get_target_seq_len_block_size() {
    constexpr size_t block_size = 16;
    return block_size;
}

inline size_t get_sg_number_scale_factor(const device_info& info, size_t head_size, size_t kernel_type) {
    const size_t optimal_scale_factor = 2;
    if (kernel_type == SDPAStage::MULTI_TOKENS) {
        if (head_size * optimal_scale_factor <= info.max_work_group_size) {
            return optimal_scale_factor;
        }
    } else if (kernel_type == SDPAStage::SINGLE_TOKEN) {
        if (head_size * optimal_scale_factor <= info.max_work_group_size &&
            head_size * optimal_scale_factor / subgroup_size <= subgroup_size) {
            return optimal_scale_factor;
        }
    }

    return 1;
}

inline size_t get_seq_len_partition_size(const device_info& info, size_t head_size, size_t kernel_type) {
    size_t seq_len = 0;
    if (kernel_type == SDPAStage::MULTI_TOKENS) {
        seq_len = head_size * get_sg_number_scale_factor(info, head_size, kernel_type);
    } else {
        seq_len = 256;
    }

    return seq_len;
}

inline size_t get_partitions_num(const kernel_impl_params& params, size_t kernel_type) {
    if (params.is_dynamic() || kernel_type == SDPAStage::MULTI_TOKENS)
        return 1;

    auto desc = params.typed_desc<scaled_dot_product_attention>();

    const auto head_size = params.input_layouts[0].get_partial_shape()[3].get_length();
    const auto source_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_k_transpose_order), params.input_layouts[1]);

    return ceil_div(source_seq_len, get_seq_len_partition_size(params.get_device_info(), head_size, kernel_type));
}

inline bool is_prefill_stage(const kernel_impl_params& params) {
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), params.input_layouts[0]);

    return target_seq_len > 1;
}

class SDPAOptGeneratorBase : public SDPABase {
public:
    SDPAOptGeneratorBase(std::string name, bool indirect) : SDPABase(name, indirect) {}

protected:
    JitConstants get_jit_constants_base(const kernel_impl_params& params, size_t stage, bool add_tensor_definitions = true) const {
        auto jit = SDPABase::get_jit_constants(params);
        if (add_tensor_definitions)
            jit.merge(make_tensors_jit_constants(params));
        const auto& info = params.get_device_info();
        const auto& q_layout = params.get_input_layout(0);

        constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
        jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", softmax_accumulator_type));
        jit.make("SUBGROUP_SIZE", subgroup_size);

        auto [broadcast_axis, group_size] = get_gqa_params(params);
        int64_t head_size = -1;
        if (params.is_type<scaled_dot_product_attention>()) {
            head_size = q_layout.get_partial_shape()[3].get_length();
        } else if (params.is_type<paged_attention>()) {
            auto desc = params.typed_desc<paged_attention>();
            head_size = desc->head_size;
        }

        jit.make("SEQ_LEN_PARTITION_SIZE", get_seq_len_partition_size(info, head_size, stage));
        jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(info, head_size, stage));

        if (params.is_type<scaled_dot_product_attention>()) {
            if (get_data_inputs_num(*params.typed_desc<scaled_dot_product_attention>()) <= 4) {
                jit.make("STATIC_SCALE_VALUE_INV", std::sqrt(static_cast<float>(head_size)));
                jit.make("STATIC_SCALE_VALUE", 1.0f / std::sqrt(static_cast<float>(head_size)));
            }
            jit.make("NUM_KV_HEADS", -1);
        } else if (params.is_type<paged_attention>()) {
            auto desc = params.typed_desc<paged_attention>();
            jit.make("IS_PAGED_ATTENTION", 1);
            jit.make("NUM_KV_HEADS", desc->kv_heads_num);
            if (desc->has_alibi)
                jit.make("HAS_ALIBI", 1);

            if (params.output_layouts.size() > 1) {
                jit.make("PAGED_ATTENTION_SCORES_OUTPUT", 1);
            }
        }

        if (info.supports_immad && broadcast_axis == -1 && head_size >= 128)
            jit.make("LOAD_KEY_LEFTOVERS_IN_CALC_LOOP", 1);

        return jit;
    }

    Arguments get_arguments_desc_impl(const kernel_impl_params& params, size_t stage) const {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

        auto desc = params.typed_desc<scaled_dot_product_attention>();

        size_t data_inputs_num = get_data_inputs_num(*desc);
        auto has_zp_input_buffers = desc->get_compression_zp_inputs_num() > 0;

        for (uint32_t i = 0; i < data_inputs_num; i++)
            args.push_back({ArgumentDescriptor::Types::INPUT, i});

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
};

class SDPAOptGeneratorSingleToken : public SDPAOptGeneratorBase {
public:
    SDPAOptGeneratorSingleToken(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", indirect) {
        m_stage_suffix = indirect ? "_single_ind" : "_single_reg";
    }

protected:
    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        return get_arguments_desc_impl(params, SDPAStage::SINGLE_TOKEN);
    }

    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::SINGLE_TOKEN);
        jit.make("SDPA_STAGE_0", 1);
        jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", 1);
        return jit;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<scaled_dot_product_attention>();

                const auto& out_l = params.output_layouts[0];
                const auto& q_l = params.input_layouts[0];

                const auto batch_size = extract_channel(get_transposed_channel(ChannelName::BATCH, desc->output_transpose_order), out_l);
                const auto heads_num = extract_channel(get_transposed_channel(ChannelName::FEATURE, desc->output_transpose_order), out_l);
                const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), q_l);
                const auto head_size = q_l.get_partial_shape()[3].get_length();
                const size_t num_of_partitions = get_partitions_num(params, SDPAStage::SINGLE_TOKEN);

                const size_t sg_num_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::SINGLE_TOKEN);
                wgs.global = { batch_size * heads_num, target_seq_len, head_size * num_of_partitions * sg_num_scale };
                wgs.local = { 1, 1, head_size * sg_num_scale };
            }
        };
        return f;
    }
};

class SDPAOptGeneratorMultiToken : public SDPAOptGeneratorBase {
public:
    SDPAOptGeneratorMultiToken(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", indirect) {
        m_stage_suffix = indirect ? "_multi_ind" : "_multi_reg";
    }
    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        return get_arguments_desc_impl(params, SDPAStage::MULTI_TOKENS);
    }

    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::MULTI_TOKENS);
        jit.make("SDPA_STAGE_0", 1);
        jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());
        return jit;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<scaled_dot_product_attention>();

                const auto& out_l = params.output_layouts[0];
                const auto& q_l = params.input_layouts[0];

                const auto batch_size = extract_channel(get_transposed_channel(ChannelName::BATCH, desc->output_transpose_order), out_l);
                const auto heads_num = extract_channel(get_transposed_channel(ChannelName::FEATURE, desc->output_transpose_order), out_l);
                const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), q_l);
                const size_t target_seq_len_block_size = get_target_seq_len_block_size();
                const auto head_size = q_l.get_partial_shape()[3].get_length();
                const size_t sg_num_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::MULTI_TOKENS);

                wgs.global = { batch_size * heads_num,
                               ceil_div(target_seq_len, target_seq_len_block_size),
                               head_size * sg_num_scale };
                wgs.local = { 1, 1, head_size * sg_num_scale };


            }
        };
        return f;
    }
};

class SDPAOptGeneratorFinalization : public SDPAOptGeneratorBase {
public:
    SDPAOptGeneratorFinalization(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", false) {
        m_stage_suffix = "_finalization";
    }
    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        return get_arguments_desc_impl(params, SDPAStage::FINALIZATION);
    }

    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::SINGLE_TOKEN);
        jit.make("SDPA_STAGE_1", 1);
        jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());
        return jit;
    }

    DispatchDataFunc get_dispatch_data_func() const override {
        static auto f = DISPATCH_DATA_FUNC(params, kd, rt_params) {
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.clear();

            ScalarDescriptor num_of_partitions_scalar;
            num_of_partitions_scalar.t = ScalarDescriptor::Types::UINT32;

            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<scaled_dot_product_attention>();

                const auto& out_l = params.output_layouts[0];
                const auto& q_l = params.input_layouts[0];

                const auto batch_size = extract_channel(get_transposed_channel(ChannelName::BATCH, desc->output_transpose_order), out_l);
                const auto heads_num = extract_channel(get_transposed_channel(ChannelName::FEATURE, desc->output_transpose_order), out_l);
                const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), q_l);
                const size_t num_of_partitions = get_partitions_num(params, SDPAStage::SINGLE_TOKEN);

                const auto head_size = static_cast<size_t>(q_l.get_partial_shape()[3].get_length());

                wgs.global = { batch_size * heads_num, target_seq_len, head_size };
                wgs.local = { 1, 1, head_size };

                num_of_partitions_scalar.v.u32 = static_cast<uint32_t>(num_of_partitions);
                scalars.push_back(num_of_partitions_scalar);
            }
        };
        return f;
    }
};

}  // namespace ov::intel_gpu::ocl
