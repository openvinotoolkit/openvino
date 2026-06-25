// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifndef ENABLE_ONEDNN_FOR_GPU
    #define ENABLE_ONEDNN_FOR_GPU 1
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "sdpa_gen_ocl.hpp"
#include "paged_attention_opt.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "openvino/core/type/float16.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "paged_attention_inst.h"
#include "paged_attention_opt.hpp"
#include "sdpa_base.hpp"
#include "../utils/kernel_generator.hpp"
// clang-format on
namespace ov::intel_gpu::ocl {
namespace {

struct sdpa_ocl_config_t {
    int subgroup_size = 0;
    int kq_sg_tile_keys = 0;
    int kq_sg_tile_queries = 0;
    int kq_sg_per_wg_keys = 0;
    int kq_sg_per_wg_queries = 0;
    int sv_sg_tile_values = 0;
    int sv_sg_tile_scores = 0;
    int sv_sg_per_wg_values = 0;
    int sv_sg_per_wg_scores = 0;

    int sg_per_wg() const {
        return kq_sg_per_wg_keys * kq_sg_per_wg_queries;
    }

    int kq_wg_tile_keys() const {
        return kq_sg_tile_keys * kq_sg_per_wg_keys;
    }

    int kq_wg_tile_queries() const {
        return kq_sg_tile_queries * kq_sg_per_wg_queries;
    }
};

size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    default:
        return 16;
    }
}

inline size_t get_d_max(size_t head_size) {
    for (size_t i = 32; i <= 1024; i *= 2) {
        if (head_size <= i) {
            return i;
        }
    }
    return head_size;
}

// Per-head-size tuned tiling for the sdpa_ocl kernel, mirroring sdpa_micro's
// choose_config_* tables. The KQ workgroup tile is kept at 128 keys with sg_per_wg = 16
// across all head sizes; only the S*V split (and, for d_max <= 32, the query tile) vary.
// Every branch's values were verified by the Phase-0 constraint checker for:
//   - sv_value_blocks >= 1 and sv_score_blocks >= 1 (non-empty DPAS tiles),
//   - WG coverage (sv tile_values*per_wg == d_max, tile_scores*per_wg == wg_queries),
//   - SLM budget and alpha-rescale (KQ/SV query-split) alignment.
// arch is currently used only for the subgroup size; per-arch specialization can be
// added here later the same way sdpa_micro splits choose_config_xehpc/xe2/xe3p.
sdpa_ocl_config_t choose_config(gpu_arch arch, size_t d_max) {
    sdpa_ocl_config_t config;  // struct defaults already encode the d_max <= 128 tiling
    config.subgroup_size = static_cast<int>(get_subgroup_size(arch));

    config.kq_sg_tile_keys = 16;
    config.kq_sg_tile_queries = 16;
    config.kq_sg_per_wg_keys = 8;
    config.kq_sg_per_wg_queries = 2;
    config.sv_sg_tile_values = 16;
    config.sv_sg_tile_scores = 16;
    config.sv_sg_per_wg_values = 8;
    config.sv_sg_per_wg_scores = 2;

    if (d_max <= 32) {
        // wg_queries is doubled to 64 so the 16 subgroups still get a valid S*V split
        // (sv_sg_tile_values must be >= 16; a 32-wide head dim cannot be split 8 ways).
        config.kq_sg_tile_queries = 32;
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 8;
        config.sv_sg_per_wg_values = 2;
        config.sv_sg_per_wg_scores = 8;
    } else if (d_max <= 64) {
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 8;
        config.sv_sg_per_wg_values = 4;
        config.sv_sg_per_wg_scores = 4;
    } else if (d_max <= 128) {
        config.sv_sg_tile_values = 16;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    } else if (d_max <= 256) {
        config.sv_sg_tile_values = 32;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    } else {
        // d_max <= 512; larger head sizes are rejected upstream by supports_micro_sdpa.
        config.sv_sg_tile_values = 64;
        config.sv_sg_tile_scores = 16;
        config.sv_sg_per_wg_values = 8;
        config.sv_sg_per_wg_scores = 2;
    }
    return config;
}

JitConstants unit_parameters(const std::string& prefix) {
    JitConstants definitions({});
    for (size_t i = 0; i < 4; i++) {
        definitions.make(prefix + "_B" + std::to_string(i), 1);
        definitions.make(prefix + "_SB" + std::to_string(i), 1);
    }

    return definitions;
}

JitConstants convert_strides(std::string target_prefix, std::string source_prefix, const std::vector<int64_t> order) {
    JitConstants definitions({});

    std::vector<std::string> target_stride_definitions = {
        target_prefix + "_S0",
        target_prefix + "_S1",
        target_prefix + "_S2",
        target_prefix + "_S3",
    };

    std::vector<std::string> source_stride_definitions = {
        source_prefix + "_BATCH_PITCH",
        source_prefix + "_FEATURE_PITCH",
        source_prefix + "_Y_PITCH",
        source_prefix + "_X_PITCH",
    };

    std::vector<std::string> target_size_definitions = {
        target_prefix + "_D0",
        target_prefix + "_D1",
        target_prefix + "_D2",
        target_prefix + "_D3",
    };

    std::vector<std::string> source_size_definitions = {
        source_prefix + "_BATCH_NUM",
        source_prefix + "_FEATURE_NUM",
        source_prefix + "_SIZE_Y",
        source_prefix + "_SIZE_X",
    };

    for (size_t i = 0; i < target_stride_definitions.size(); i++) {
        definitions.make(target_stride_definitions[i], source_stride_definitions[order[i]]);
        definitions.make(target_size_definitions[i], source_size_definitions[order[i]]);
    }

    return definitions;
}

inline size_t micro_get_num_heads(const kernel_impl_params& params, size_t qkv_idx) {
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        switch (qkv_idx) {
        case 0:
            return desc->heads_num;
        case 1:
            return desc->kv_heads_num;
        case 2:
            return desc->kv_heads_num;
        default:
            OPENVINO_THROW("Invalid qkv index for paged attention");
        }
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0:
            return get_num_heads(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));
        case 1:
            return get_num_heads(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
        case 2:
            return get_num_heads(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    return -1;
}

inline size_t micro_get_head_size(const kernel_impl_params& params, size_t qkv_idx) {
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        switch (qkv_idx) {
        case 0:
            return desc->k_head_size;
        case 1:
            return desc->k_head_size;
        case 2:
            return desc->v_head_size;
        default:
            OPENVINO_THROW("Invalid qkv index for paged attention");
        }
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0:
            return get_head_size(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));
        case 1:
            return get_head_size(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
        case 2:
            return get_head_size(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    return -1;
}

inline ov::Dimension micro_get_seq_length(const kernel_impl_params& params, int32_t qkv_idx) {
    if (qkv_idx < 0 || qkv_idx > 2) {
        OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
    }
    if (params.is_type<paged_attention>()) {
        return ov::Dimension(params.input_layouts[qkv_idx].get_partial_shape()[0]);
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0:
            return get_seq_length(params.input_layouts[0], extend_order_in_num_heads_dim(desc->input_q_transpose_order));
        case 1:
            return get_seq_length(params.input_layouts[1], extend_order_in_num_heads_dim(desc->input_k_transpose_order));
        case 2:
            return get_seq_length(params.input_layouts[2], extend_order_in_num_heads_dim(desc->input_v_transpose_order));
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    return ov::Dimension();
}

inline ov::Dimension micro_get_aligned_seq_length(const kernel_impl_params& params, int32_t qkv_idx, int64_t target_seq_len_block_size = 16) {
    if (qkv_idx < 0 || qkv_idx > 2) {
        OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
    }
    if (params.is_type<paged_attention>()) {
        const auto desc = params.typed_desc<paged_attention>();
        const auto& input_mem = params.memory_deps;
        const auto subsequence_begins_mem = input_mem.at(paged_attention::PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *params.strm);
        auto aligned_seq_len = 0;
        for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
            auto prompt_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];
            aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
        }
        return aligned_seq_len;
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        switch (qkv_idx) {
        case 0:
            return get_seq_length(params.input_layouts[0], desc->input_q_transpose_order);
        case 1:
            return get_seq_length(params.input_layouts[1], desc->input_k_transpose_order);
        case 2:
            return get_seq_length(params.input_layouts[2], desc->input_v_transpose_order);
        default:
            OPENVINO_THROW("Invalid qkv index for scaled dot product attention");
        }
    }
    return ov::Dimension();
}

inline size_t micro_get_input_num(const kernel_impl_params& params, const sdpa_configuration& config) {
    auto data_inputs_num = config.input_num;
    bool is_paged_attention = params.is_type<paged_attention>() ? true : false;
    if (!is_paged_attention) {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        data_inputs_num = get_data_inputs_num(*desc);
    }
    return data_inputs_num;
}

}  // namespace

std::string SDPAOclGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";

    return base_options + extra_options;
}

void SDPAOclGenerator::init_sdpa_configuration(const kernel_impl_params& impl_param, sdpa_configuration& sdpa_config) {
    if (impl_param.is_type<scaled_dot_product_attention>()) {
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

        sdpa_config = get_sdpa_configuration(impl_param, extended_input_q_transpose_order, extended_input_k_transpose_order, extended_input_v_transpose_order);
    } else {
        bool is_dynamic = impl_param.is_dynamic();
        const auto desc = impl_param.typed_desc<paged_attention>();
        sdpa_config.k_head_size = desc->k_head_size;
        sdpa_config.v_head_size = desc->v_head_size;
        sdpa_config.heads_num = desc->heads_num;
        sdpa_config.kv_heads_num = desc->kv_heads_num;
        sdpa_config.has_alibi_input = desc->has_alibi;
        sdpa_config.is_causal = true;
        sdpa_config.is_paged_attention = true;
        sdpa_config.paged_attention_block_size = static_cast<int64_t>(paged_attention::block_size);
        sdpa_config.paged_attention_sliding_window = desc->sliding_window;
        sdpa_config.has_score_aggregation = desc->has_score_aggregation;

        if (desc->scale_val.has_value()) {
            sdpa_config.has_const_scale_val = true;
            sdpa_config.scale_val = desc->scale_val.value();
        } else {
            sdpa_config.has_const_scale_val = false;
        }

        sdpa_config.has_score_aggregation = desc->has_score_aggregation;
        sdpa_config.has_rotated_blocks = desc->has_rotated_blocks;

        if (desc->heads_num != desc->kv_heads_num) {
            sdpa_config.broadcast_axis = 1;
            sdpa_config.kv_group_size = desc->heads_num / desc->kv_heads_num;
        }

        if (desc->has_scores_output() && !is_dynamic) {
            const auto& input_mem = impl_param.memory_deps;
            const auto max_context_len = input_mem.at(12);  // PagedAttentionInputIdx::MAX_CONTEXT_LEN
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *impl_param.strm);
            sdpa_config.paged_attention_max_len = max_context_len_mem_lock[0];

            if (desc->has_score_aggregation) {
                const auto score_aggregation = input_mem.at(13);  // PagedAttentionInputIdx::SCORE_AGGREGATION
                mem_lock<int32_t, mem_lock_type::read> score_aggregation_mem_lock(score_aggregation, *impl_param.strm);

                auto total_tokens_num = 0;
                for (size_t i = 0; i < score_aggregation_mem_lock.size(); i++) {
                    total_tokens_num += score_aggregation_mem_lock[i];
                }
                sdpa_config.paged_attention_snap_kv_tokens = total_tokens_num;
            }
        }

        // If micro sdpa kernel is called by paged attention, then it is always used for prefill stage, and compressed QKV is not used.
        sdpa_config.is_kv_compressed = false;
        sdpa_config.use_asymmetric_quantization = false;

        // PagedAttentionInputIdx::ALIBI
        const auto has_alibi = impl_param.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        sdpa_config.input_num = 7;
        if (has_scale_input)
            sdpa_config.input_num++;

        if (has_alibi)
            sdpa_config.input_num++;
    }
}

// Use 'maybe_unused' to avoid DPC++ build error
[[maybe_unused]] const bool kq_common_scales = false;
[[maybe_unused]] const bool kq_common_zp = false;
[[maybe_unused]] const bool vs_common_scales = false;
[[maybe_unused]] const bool vs_common_zp = false;

JitConstants SDPAOclGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(params);
    sdpa_configuration config;
    init_sdpa_configuration(params, config);

    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    jit.add(make_tensors_jit_constants(params));
    if (desc->has_sink_input) {
        const auto& sink_layout = params.input_layouts[ScaledDotProductAttentionInputIdx::SINK];
        jit.make("SINK_DATA_T", to_ocl_type(sink_layout.data_type));
        jit.make("HAS_SINK_INPUT", 1);
    }

    // QQ_BIAS is a paged-attention-only feature.
    jit.make("HAS_QQ_BIAS", 0);
    const auto& device_info = params.get_device_info();

    const auto& Q = params.input_layouts[0];
    const auto& K = (config.is_paged_attention && !m_is_prefill) ? params.input_layouts[3] : params.input_layouts[1];
    const auto& V = (config.is_paged_attention && !m_is_prefill) ? params.input_layouts[4] : params.input_layouts[2];
    const auto& out = params.output_layouts[0];
    const auto& out_ps = out.get_partial_shape();

    const auto head_size = micro_get_head_size(params, 0);
    const auto k_head_size = micro_get_head_size(params, 1);
    const auto v_head_size = micro_get_head_size(params, 2);

    const auto d_max = get_d_max(k_head_size);
    const auto batch = out_ps[0] * out_ps[1];

    auto ldq = k_head_size * ov::element::Type(Q.data_type).size();
    auto ldk = k_head_size * ov::element::Type(K.data_type).size();
    auto ldv = v_head_size * ov::element::Type(V.data_type).size();
    auto lda = v_head_size * ov::element::Type(out.data_type).size();

    const auto ocl_config = choose_config(device_info.arch, d_max);

    jit.make("DPAS_K", 16);          // intel_sub_group_f16_f16_matrix_mad_k16 only supports KSTEP of 16
    jit.make("DPAS_ROWS", 8);
    jit.make("kq_sg_tile_keys", ocl_config.kq_sg_tile_keys);
    jit.make("kq_sg_tile_queries", ocl_config.kq_sg_tile_queries);
    jit.make("kq_sg_per_wg_keys", ocl_config.kq_sg_per_wg_keys);
    jit.make("kq_sg_per_wg_queries", ocl_config.kq_sg_per_wg_queries);
    jit.make("sv_sg_tile_scores", ocl_config.sv_sg_tile_scores);
    jit.make("sv_sg_tile_values", ocl_config.sv_sg_tile_values);
    jit.make("sv_sg_per_wg_scores", ocl_config.sv_sg_per_wg_scores);
    jit.make("sv_sg_per_wg_values", ocl_config.sv_sg_per_wg_values);
    jit.make("D_MAX", d_max);
    jit.make("DKS", "(D_MAX / DPAS_K)");
    jit.make("Q_DWORDS", 8);        // 16 half values per Q KSTEP packed as 8 uint dwords.
    jit.make("SUBGROUP_SIZE", ocl_config.subgroup_size);
    jit.make("INVERT_SCALE", false);
    jit.make("SCALE_DATA_T", "half");
    jit.make("HEAD_SIZE", k_head_size);

    auto data_inputs_num = micro_get_input_num(params, config);

    size_t scale_input_idx = 4;
    jit.make("IS_CAUSAL", config.is_causal);
    if (!config.is_paged_attention) {
        const bool has_attn_mask_input = sdpa_has_runtime_attn_mask_input(params);
        if (config.has_const_attn_mask_val) {
            jit.make("WITH_ATTN_MASK", 0);
            jit.make("STATIC_SCALAR_ATTN_MASK_VALUE", config.attn_mask_val);
            // scale_input_idx -= 1;
        } else {
            jit.make("WITH_ATTN_MASK", has_attn_mask_input ? 1 : 0);
        }
        // Compile-time mask-kind specialization to drop the per-element runtime branch
        // over MSK_D2/MSK_D3 (which are shape_info-driven and thus runtime for dynamic
        // shapes). The runtime branch blocks IGC optimization across the whole hot loop;
        // pinning the kind at JIT time recovers a large amount of cmp/control-flow.
        //   2 = full 2D (query>1 & key>1), 1 = per-key (query==1 & key>1),
        //   0 = scalar/broadcast, -1 = unknown -> keep the runtime branch.
        int mask_kind = -1;
        if (has_attn_mask_input && !config.has_const_attn_mask_val) {
            const auto& msk_ps = params.input_layouts[ScaledDotProductAttentionInputIdx::ATTN_MASK].get_partial_shape();
            const auto r = msk_ps.size();
            if (r >= 2) {
                const auto& dq = msk_ps[r - 2];  // mask query dim
                const auto& dk = msk_ps[r - 1];  // mask key dim
                if (dq.is_static() && dk.is_static()) {
                    // Exact classification when both trailing dims are known.
                    const bool q_gt1 = dq.get_length() > 1;
                    const bool k_gt1 = dk.get_length() > 1;
                    mask_kind = (q_gt1 && k_gt1) ? 2 : (!q_gt1 && k_gt1) ? 1 : 0;
                } else if (dq.is_static()) {
                    const bool q_gt1 = dq.get_length() > 1;
                    mask_kind = q_gt1 ? 2 : 1;
                } else {
                    // Dynamic trailing dims: infer from the SDPA stage. Prefill processes
                    // many query tokens at once, so the mask is full 2D [.,.,q>1,k>1]
                    // (kind 2). The generate/single-token stage always has query dim == 1
                    // with a per-key mask [B,H,1,K] (kind 1). Both stages are compiled as
                    // separate kernels (regular_micro_multi_tokens / _single_token), each
                    // getting the right specialization here.
                    mask_kind = m_is_prefill ? 2 : 1;
                }
            }
        }
        jit.make("MASK_KIND", mask_kind);
    } else {
        jit.make("WITH_ATTN_MASK", 0);
        jit.make("MASK_KIND", -1);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", config.paged_attention_block_size);
    }

    if (config.has_const_scale_val) {
        jit.make("STATIC_SCALE_VALUE", config.scale_val);
        jit.make("STATIC_SCALE_VALUE_INV", 1.0f / config.scale_val);
    } else {
        jit.make("WITH_SCALE", data_inputs_num > scale_input_idx);
    }

    jit.make("Q_ALIGN", micro::alignment_for_ld(static_cast<int>(ldq)));
    jit.make("K_ALIGN", micro::alignment_for_ld(static_cast<int>(ldk)));
    jit.make("V_ALIGN", micro::alignment_for_ld(static_cast<int>(ldv)));
    jit.make("A_ALIGN", micro::alignment_for_ld(static_cast<int>(lda)));

    jit.make("IS_PREFILL", m_is_prefill);
    jit.make("TRANSPOSE_K", false);
    jit.make("IS_PAGED_ATTENTION", config.is_paged_attention ? 1 : 0);
    jit.make("KV_HEADS_NUM", config.kv_heads_num);
    jit.make("HEADS_NUM", config.heads_num);

    const auto q_heads_num = micro_get_num_heads(params, 0);
    const auto k_heads_num = micro_get_num_heads(params, 1);
    jit.make("KV_GROUP_SIZE", q_heads_num / k_heads_num);

    jit.make("QRY_DATA_T", to_ocl_type(Q.data_type));
    jit.make("KEY_DATA_T", to_ocl_type(K.data_type));
    jit.make("VAL_DATA_T", to_ocl_type(V.data_type));

    auto elems_per_byte = [](ov::element::Type dt) {
        switch (dt) {
        case ov::element::u4:
        case ov::element::i4:
            return 2;
        default:
            return 1;
        }
    };

    const bool use_asymmetric_quantization = config.use_asymmetric_quantization;
    if (!config.is_paged_attention && config.is_kv_compressed) {
        const auto& key_cache_comp_scale = params.input_layouts[data_inputs_num];
        const auto& value_cache_comp_scale = params.input_layouts[data_inputs_num + 1];
        jit.make("KV_COMPRESSED", 1);
        jit.make("KEY_ATTR_SCALES_DATA_T", to_ocl_type(key_cache_comp_scale.data_type));
        jit.make("VAL_ATTR_SCALES_DATA_T", to_ocl_type(value_cache_comp_scale.data_type));

        int kq_scale_mask = (static_cast<int>(config.is_kv_compressed) << 1) | static_cast<int>(kq_common_scales);
        int vs_scale_mask = (static_cast<int>(config.is_kv_compressed) << 1) | static_cast<int>(vs_common_scales);
        jit.make("KEY_SCALES", kq_scale_mask);
        jit.make("VAL_SCALES", vs_scale_mask);
        jit.make("KEY_GROUP_SIZE", head_size);
        jit.make("VAL_GROUP_SIZE", head_size);

        jit.add(make_layout_jit_constants("KEY_SCALE", key_cache_comp_scale, params.in_port_to_shape_info_offset.at(data_inputs_num)));
        jit.add(make_layout_jit_constants("VAL_SCALE", value_cache_comp_scale, params.in_port_to_shape_info_offset.at(data_inputs_num + 1)));

        const std::vector<int64_t> default_order = {0, 1, 2, 3};
        jit.add(convert_strides("KEY_COMP", "KEY_SCALE", default_order));
        jit.add(convert_strides("VAL_COMP", "VAL_SCALE", default_order));

        jit.add(unit_parameters("KEY_COMP"));
        jit.add(unit_parameters("VAL_COMP"));

        if (use_asymmetric_quantization) {
            const auto& key_cache_comp_zp = params.input_layouts[data_inputs_num + 2];
            const auto& value_cache_comp_zp = params.input_layouts[data_inputs_num + 3];
            jit.make("KEY_ATTR_ZP_DATA_T", to_ocl_type(key_cache_comp_zp.data_type));
            jit.make("VAL_ATTR_ZP_DATA_T", to_ocl_type(value_cache_comp_zp.data_type));

            int kq_zp_mask = (static_cast<int>(use_asymmetric_quantization) << 1) | static_cast<int>(kq_common_zp);
            int vs_zp_mask = (static_cast<int>(use_asymmetric_quantization) << 1) | static_cast<int>(vs_common_zp);
            jit.make("KEY_ZERO_POINTS", kq_zp_mask);
            jit.make("VAL_ZERO_POINTS", vs_zp_mask);
            jit.make("KEY_ZP_ELEMENTS_PER_BYTE", elems_per_byte(key_cache_comp_zp.data_type));
            jit.make("VAL_ZP_ELEMENTS_PER_BYTE", elems_per_byte(value_cache_comp_zp.data_type));
        }
    }

    if (config.is_paged_attention && data_type_traits::is_i8_u8(K.data_type)) {
        auto pa_desc = params.typed_desc<paged_attention>();
        jit.make("IS_KV_COMPRESSED_PA", true);

        const auto kv_precision = params.get_program().get_config().get_kv_cache_precision();
        const bool is_int4_logical = data_type_traits::is_i4_u4(kv_precision);

        auto scales_zp_size = 4;  // scale + zp
        if (is_int4_logical) {
            // INT4 KV cache with BY_CHANNEL K quantization:
            // K: dim order {0,1,3,2} (col-major), packed block_size in innermost dim
            //    physical: [blocks, heads, head_size, packed_block + scales] u8
            //    packed_block = block_size/2 bytes, scales = 4 bytes
            // V: dim order {0,1,2,3} (row-major), packed head_size in innermost dim
            //    physical: [blocks, heads, block_size, packed_head + scales] u8
            jit.make("IS_INT4_KV_CACHE", 1);
            jit.make("IS_KEY_BY_CHANNEL", 1);
            jit.make("ADJUSTED_K_HEAD_SIZE", k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", v_head_size / 2 + scales_zp_size);
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", config.paged_attention_block_size / 2 + scales_zp_size);
        } else if (pa_desc->is_key_by_channel) {
            jit.make("IS_KEY_BY_CHANNEL", 1);
            jit.make("ADJUSTED_K_HEAD_SIZE", k_head_size);
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", config.paged_attention_block_size + scales_zp_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", v_head_size + scales_zp_size);
        } else {
            jit.make("ADJUSTED_K_HEAD_SIZE", k_head_size + scales_zp_size);
            jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", config.paged_attention_block_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", v_head_size + scales_zp_size);
        }
    } else if (config.is_paged_attention) {
        jit.make("ADJUSTED_K_HEAD_SIZE", k_head_size);
        jit.make("ADJUSTED_V_HEAD_SIZE", v_head_size);
        jit.make("ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE", config.paged_attention_block_size);
    }

    jit.make("KEY_ELEMENTS_PER_BYTE", elems_per_byte(params.input_layouts[1].data_type));
    jit.make("VAL_ELEMENTS_PER_BYTE", elems_per_byte(params.input_layouts[2].data_type));


    auto convert_strides = [](std::string target_prefix, std::string source_prefix, const std::vector<int64_t> order) {
        JitConstants definitions({});

        std::vector<std::string> target_stride_definitions = {
            target_prefix + "_S0",
            target_prefix + "_S1",
            target_prefix + "_S2",
            target_prefix + "_S3",
        };

        std::vector<std::string> source_stride_definitions = {
            source_prefix + "_BATCH_PITCH",
            source_prefix + "_FEATURE_PITCH",
            source_prefix + "_Y_PITCH",
            source_prefix + "_X_PITCH",
        };

        std::vector<std::string> target_size_definitions = {
            target_prefix + "_D0",
            target_prefix + "_D1",
            target_prefix + "_D2",
            target_prefix + "_D3",
        };

        std::vector<std::string> source_size_definitions = {
            source_prefix + "_BATCH_NUM",
            source_prefix + "_FEATURE_NUM",
            source_prefix + "_SIZE_Y",
            source_prefix + "_SIZE_X",
        };

        for (size_t i = 0; i < target_stride_definitions.size(); i++) {
            definitions.make(target_stride_definitions[i], source_stride_definitions[order[i]]);
            definitions.make(target_size_definitions[i], source_size_definitions[order[i]]);
        }

        return definitions;
    };

    if (config.is_paged_attention) {
        const std::vector<int64_t> default_order = {0, 1, 2, 3};
        jit.add(convert_strides("QRY", "INPUT0", default_order));
        jit.add(convert_strides("KEY", "INPUT1", default_order));
        jit.add(convert_strides("VAL", "INPUT2", default_order));
        jit.add(convert_strides("DST", "OUTPUT", default_order));

    } else {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);
        jit.add(convert_strides("QRY", "INPUT0", extended_input_q_transpose_order));
        jit.add(convert_strides("KEY", "INPUT1", extended_input_k_transpose_order));
        jit.add(convert_strides("VAL", "INPUT2", extended_input_v_transpose_order));
        jit.add(convert_strides("DST", "OUTPUT", extended_output_transpose_order));
    }

    jit.add(unit_parameters("QRY"));
    jit.add(unit_parameters("KEY"));
    jit.add(unit_parameters("VAL"));
    jit.add(unit_parameters("DST"));

    if (data_inputs_num > 3 && !config.is_paged_attention && sdpa_has_runtime_attn_mask_input(params)) {
        jit.add(convert_strides("MSK", "INPUT3", {0, 1, 2, 3}));
        jit.add(unit_parameters("MSK"));
    }

    // std::cout << "JIT for micro kernel:" << std::endl;
    // for (auto it : jit) {
    //     std::cout << "jit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;

    return jit;
}

Arguments SDPAOclGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    sdpa_configuration config;
    init_sdpa_configuration(params, config);
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    auto data_inputs_num = micro_get_input_num(params, config);

    if (config.is_paged_attention) {
        const auto desc = params.typed_desc<paged_attention>();
        const auto has_qq_bias = desc->has_qq_bias;
        if (m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // Key
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // Q
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // Value
        } else {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // Key cache
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // Q
            args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // Value cache
            args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // Key
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // Value
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // A

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        if (!m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
            args.push_back({ArgumentDescriptor::Types::INPUT, 7});  // block_indices
            args.push_back({ArgumentDescriptor::Types::INPUT, 8});  // block_indices_begins
        }
        if (!config.has_const_scale_val)
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});  // scale

        if (desc->has_sink_input)
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SINKS});  // sink

        if (has_qq_bias && !m_is_prefill) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QQ_BIAS});  // qq_bias
            args.push_back(
                {ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QQ_BIAS_BEGINS});  // qq_bias_begins                              // qq_bias_num
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // blocked_indexes_start_and_gws_mapping
    } else {
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::KEY});    // K
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::QUERY});  // Q
        args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::VALUE});  // V
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                                        // A

        const uint32_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
        if (sdpa_has_runtime_attn_mask_input(params))
            args.push_back({ArgumentDescriptor::Types::INPUT, attn_mask_idx});  // mask
        const uint32_t scale_idx = ScaledDotProductAttentionInputIdx::SCALE;
        if (config.input_num > scale_idx && !config.has_const_scale_val)
            args.push_back({ArgumentDescriptor::Types::INPUT, scale_idx});  // Scale
        const uint32_t sink_idx = ScaledDotProductAttentionInputIdx::SINK;
        if (config.input_num > sink_idx)
            args.push_back({ArgumentDescriptor::Types::INPUT, sink_idx});  // Sink

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // D
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // K
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // Q
        // args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // scale
    }

    if (config.is_kv_compressed) {
        const bool is_asym_quantization = config.use_asymmetric_quantization;
        uint32_t input_idx = static_cast<uint32_t>(data_inputs_num);
        args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 0});  // K scales
        if (is_asym_quantization)
            args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 2});  // K zp

        args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 1});  // V scales
        if (is_asym_quantization)
            args.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 3});  // V zp
    }

    return args;
}

DispatchDataFunc SDPAOclGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(3);

        auto params = impl_param;
        if (!params.is_dynamic()) {
            const auto& out = params.output_layouts[0];
            const auto& out_ps = out.get_partial_shape();

            const auto& device_info = params.get_device_info();
            const auto k_head_size = micro_get_head_size(params, 1);
            const auto d_max = get_d_max(k_head_size);
            const auto ocl_config = choose_config(device_info.arch, d_max);

            const ov::Dimension n_keys = micro_get_aligned_seq_length(params, 1, ocl_config.kq_wg_tile_keys());
            const ov::Dimension n_queries = micro_get_aligned_seq_length(params, 0, ocl_config.kq_wg_tile_queries());
            const auto v_head_size = micro_get_head_size(params, 2);

            size_t q = n_queries.get_length();

            wgs.local = {static_cast<size_t>(ocl_config.subgroup_size), static_cast<size_t>(ocl_config.sg_per_wg()), 1};
            wgs.global = wgs.local;
            wgs.global[0] = wgs.global[0] * ((q + ocl_config.kq_wg_tile_queries() - 1) / ocl_config.kq_wg_tile_queries());
            wgs.global[1] *= out_ps[1].get_length();
            wgs.global[2] *= out_ps[0].get_length();

            auto to_int32 = [](size_t value) {
                if (value > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    return static_cast<int32_t>(-1);
                }
                return static_cast<int32_t>(value);
            };

            ScalarDescriptor s_d{ScalarDescriptor::Types::INT32};
            s_d.v.s32 = to_int32(v_head_size);
            scalars.push_back(s_d);

            ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
            s_k.v.s32 = to_int32(n_keys.get_length());
            scalars.push_back(s_k);

            ScalarDescriptor s_q{ScalarDescriptor::Types::INT32};
            s_q.v.s32 = to_int32(n_queries.get_length());
            scalars.push_back(s_q);

            // ScalarDescriptor s_scale{ScalarDescriptor::Types::FLOAT32};
            // s_scale.v.f32 = static_cast<float>(1.0f / std::sqrt(static_cast<float>(v_head_size)));
            // scalars.push_back(s_scale);
        }
    }};
}

}  // namespace ov::intel_gpu::ocl
#endif
