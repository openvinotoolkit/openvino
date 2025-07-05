// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_opt.hpp"
#include "utils/kernel_generator.hpp"

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
        if (head_size * optimal_scale_factor <= info.max_work_group_size) {
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
        seq_len = head_size * get_sg_number_scale_factor(info, head_size, kernel_type);
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

    const auto head_size = params.input_layouts[0].get_partial_shape()[3].get_length();
    const auto source_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_k_transpose_order), params.input_layouts[1]);

    return ceil_div(source_seq_len, get_seq_len_partition_size(params.get_device_info(), head_size, kernel_type));
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
