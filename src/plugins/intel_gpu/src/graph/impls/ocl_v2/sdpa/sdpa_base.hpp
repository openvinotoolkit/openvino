// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/type.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_utils.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct sdpa_configuration {
    int64_t k_head_size = -1;
    int64_t v_head_size = -1;
    int64_t heads_num = -1;
    int64_t kv_heads_num = -1;

    // GQA configuration
    int64_t kv_group_size = 1;
    int64_t broadcast_axis = -1;

    bool is_causal = false;
    bool has_alibi_input = false;
    bool is_kv_compressed = false;
    bool use_asymmetric_quantization = false;
    bool combine_scales_and_zp = false;
    bool per_head_quantization = false;

    // Paged Attention configuration
    bool is_paged_attention = false;
    size_t paged_attention_sliding_window = 0;
    int64_t paged_attention_block_size = 0;

    // Runtime Paged Attention params
    int64_t paged_attention_aligned_seq_len = -1;
    int64_t paged_attention_max_len = 0;
    int64_t paged_attention_snap_kv_tokens = 0;

    bool has_const_scale_val = false;
    float scale_val = 0.f;
    bool has_const_attn_mask_val = false;
    float attn_mask_val = 0.f;
    bool has_score_aggregation = false;
    bool has_rotated_blocks = false;

    int64_t input_num;
};

struct SDPABase : public KernelGenerator {
    SDPABase(std::string_view name, std::string_view suffix, bool indirect) : KernelGenerator(name, suffix), m_indirect(indirect) {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override;
    [[nodiscard]] std::pair<int64_t, int64_t> get_gqa_params(const kernel_impl_params& params) const;

    static sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param,
                                                     const std::vector<int64_t>& input_q_transpose_order,
                                                     const std::vector<int64_t>& input_k_transpose_order,
                                                     const std::vector<int64_t>& input_v_transpose_order);

    static bool requires_shape_canonicalization(const kernel_impl_params& impl_params);
    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params);

    bool m_indirect;
};

class SDPAImplBase : public PrimitiveImplOCL {
public:
    explicit SDPAImplBase(const std::string& name) : PrimitiveImplOCL(name) {}
    explicit SDPAImplBase(const ov::DiscreteTypeInfo& info) : PrimitiveImplOCL(std::string(info.name)) {}

    void update(cldnn::primitive_inst& inst, const RuntimeParams& impl_params) override;

    static size_t get_beam_table_id(const std::shared_ptr<const scaled_dot_product_attention>& primitive) {
        return primitive->input_size() - 1;
    }

    static bool need_indirect_load(const scaled_dot_product_attention_inst& instance) {
        auto desc = instance.get_typed_desc<scaled_dot_product_attention>();

        if (!instance.has_indirect_inputs()) {
            return false;
        }

        const auto& params = *instance.get_impl_params();
        const auto indirect_axis = desc->indirect_axis;
        if (params.input_layouts[get_beam_table_id(desc)].get_partial_shape()[indirect_axis].get_length() == 1) {
            return false;
        }

        const auto& deps = instance.dependencies();

        const auto indirect_dep_idx = 1;
        const auto& indirect_dep = deps[indirect_dep_idx].first;
        if (dynamic_cast<const kv_cache_inst*>(indirect_dep) == nullptr) {
            return true;
        }

        auto state_layout = indirect_dep->get_impl_params()->get_input_layout(0);
        bool is_prefill = state_layout.count() == 0;
        return !is_prefill;
    }
};

}  // namespace ov::intel_gpu::ocl
