// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_opt.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

inline size_t get_target_seq_len_block_size() {
    constexpr size_t block_size = 16;
    return block_size;
}

inline size_t get_sg_number_scale_factor(const device_info& info, size_t head_size, size_t kernel_type) {
    constexpr size_t subgroup_size = 16;
    const size_t optimal_scale_factor = 2;
    if (kernel_type == SDPAStage::MULTI_TOKENS) {
        if ((head_size % subgroup_size == 0) && (head_size * optimal_scale_factor <= info.max_work_group_size)) {
            return optimal_scale_factor;
        }
    } else if (kernel_type == SDPAStage::SINGLE_TOKEN) {
        if (head_size * optimal_scale_factor <= info.max_work_group_size && head_size * optimal_scale_factor / subgroup_size <= subgroup_size) {
            return optimal_scale_factor;
        }
    }

    return 1;
}

inline size_t get_seq_len_partition_size(const device_info& info, size_t head_size, size_t kernel_type) {
    size_t seq_len = 0;
    if (kernel_type == SDPAStage::MULTI_TOKENS) {
        seq_len = align_to(head_size * get_sg_number_scale_factor(info, head_size, kernel_type), 16);
    } else {
        seq_len = 256;
    }

    return seq_len;
}

inline size_t get_partitions_num(const kernel_impl_params& params, size_t kernel_type) {
    if (params.is_dynamic() || kernel_type == SDPAStage::MULTI_TOKENS) {
        return 1;
    }

    auto desc = params.typed_desc<scaled_dot_product_attention>();

    auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
    auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
    const auto head_size = get_head_size(params.get_input_layout(0), extended_input_q_transpose_order);
    const auto source_seq_len = get_seq_length(params.get_input_layout(1), extended_input_k_transpose_order);

    return ceil_div(source_seq_len, get_seq_len_partition_size(params.get_device_info(), head_size, kernel_type));
}

// Query heads per KV head that the single-token kernel processes as rows of one work-group, so that each K/V
// element is loaded once for all of them; 1 when the SDPA is not GQA or the path does not support it.
inline size_t get_gqa_single_token_rows(const kernel_impl_params& params, bool indirect) {
    if (indirect || !params.is_type<scaled_dot_product_attention>()) {
        return 1;
    }
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    const auto q_heads = get_num_heads(params.get_input_layout(0), extend_order_in_num_heads_dim(desc->input_q_transpose_order));
    const auto k_heads = get_num_heads(params.get_input_layout(1), extend_order_in_num_heads_dim(desc->input_k_transpose_order));
    const auto k_dt = params.get_input_layout(1).data_type;
    if (q_heads <= 0 || k_heads <= 0 || q_heads <= k_heads || q_heads % k_heads != 0 || data_type_traits::is_i4_u4(k_dt)) {
        return 1;
    }
    constexpr int64_t max_rows = 8;
    return q_heads / k_heads <= max_rows ? static_cast<size_t>(q_heads / k_heads) : 1;
}

// Symmetric int8 KV with one scale per channel ([B, H, 1, head_size] scale input): the single-token kernel folds
// the key scales into Q and applies the value scales to its output. Returns {keys, values}.
inline std::pair<bool, bool> get_single_token_per_channel_kv_scales(const kernel_impl_params& params) {
    if (!params.is_type<scaled_dot_product_attention>()) {
        return {false, false};
    }
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (!desc->is_kv_compressed || SDPABase::is_int4_kv_cache(params) ||
        desc->quantization_attributes.quantization_type != ov::op::internal::DynamicQuantize::QuantizationType::Symmetric) {
        return {false, false};
    }
    const auto data_inputs_num = get_data_inputs_num(*desc);
    const auto is_per_channel = [&params](size_t idx) {
        const auto& shape = params.input_layouts[idx].get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 4 && shape[2].is_static() && shape[2].get_length() == 1 && shape[3].is_static() &&
               shape[3].get_length() > 1;
    };
    return {is_per_channel(data_inputs_num), is_per_channel(data_inputs_num + 1)};
}

class SDPAOptGeneratorBase : public SDPABase {
public:
    SDPAOptGeneratorBase(std::string_view name, std::string_view stage, bool indirect) : SDPABase(name, stage, indirect) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants_base(const kernel_impl_params& params, size_t stage, bool add_tensor_definitions = true) const;
    [[nodiscard]] Arguments get_arguments_desc_impl(const kernel_impl_params& params, size_t stage) const;
};

class SDPAOptGeneratorSingleToken : public SDPAOptGeneratorBase {
public:
    explicit SDPAOptGeneratorSingleToken(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", indirect ? "_single_ind" : "_single_reg", indirect) {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class SDPAOptGeneratorMultiToken : public SDPAOptGeneratorBase {
public:
    explicit SDPAOptGeneratorMultiToken(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", indirect ? "_multi_ind" : "_multi_reg", indirect) {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

class SDPAOptGeneratorFinalization : public SDPAOptGeneratorBase {
public:
    explicit SDPAOptGeneratorFinalization(bool indirect) : SDPAOptGeneratorBase("sdpa_opt", "_finalization", false) {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

}  // namespace ov::intel_gpu::ocl
