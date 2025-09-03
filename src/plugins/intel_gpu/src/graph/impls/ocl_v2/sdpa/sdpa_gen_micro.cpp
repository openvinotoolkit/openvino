// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "sdpa_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "paged_attention_inst.h"
#include "sdpa_base.hpp"
#include "../utils/kernel_generator.hpp"
// clang-format on
namespace ov::intel_gpu::ocl {
namespace {
size_t get_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    case gpu_arch::xe_hpc:
    case gpu_arch::xe2:
    case gpu_arch::xe3:
        return 16;
    default:
        return 0;
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

micro::Type convert_type(ov::element::Type t) {
    switch (t) {
    case ov::element::f32:
        return micro::Type::f32;
    case ov::element::f16:
        return micro::Type::f16;
    case ov::element::i8:
        return micro::Type::s8;
    case ov::element::u8:
        return micro::Type::u8;
    case ov::element::i32:
        return micro::Type::s32;
    default:
        break;
    }
    OPENVINO_THROW("Unsupported element type: ", t);
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

inline size_t micro_get_key_cache_id(const kernel_impl_params& params) {
    if (params.is_type<paged_attention>()) {
        const size_t key_cache_id = 3;  // Key cache inputs
        return key_cache_id;
    } else {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        return get_key_cache_id(*desc);
    }
}

inline size_t micro_get_value_cache_id(const kernel_impl_params& params) {
    if (params.is_type<paged_attention>()) {
        const size_t value_cache_id = 4;  // Value cache inputs
        return value_cache_id;
    } else {
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        return get_value_cache_id(*desc);
    }
}

struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq;  // Subgroup tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs;  // Subgroup tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq;          // Workgroup configuration for K*Q GEMM
    int wg_m_vs, wg_n_vs;          // Workgroup configuration for V*S GEMM
};

// Kernel configurations:
//  h<N> -- maximum head size = N
//  s<M> -- target sequence length = M
//   2nd -- second token (thin Q)
sdpa_config_t xehpg_h32 = {32, 16, 16, 16, 2, 16, 2, 16};
sdpa_config_t xehpg_h32_s256 = {16, 16, 16, 16, 2, 8, 2, 8};
sdpa_config_t xehpg_h32_s64 = {16, 16, 16, 8, 4, 4, 2, 8};
sdpa_config_t xehpg_h32_s32 = {8, 8, 8, 8, 4, 4, 4, 4};
sdpa_config_t xehpg_h32_2nd = {8, 32, 16, 8, 8, 1, 2, 4};

sdpa_config_t xehpg_q_h32 = {32, 16, 16, 16, 2, 8, 2, 8};
sdpa_config_t xehpg_q_h32_2nd = {32, 16, 8, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_h64 = {32, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_h64_s128 = {16, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_h64_s64 = {32, 16, 16, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_h64_2nd = {8, 16, 16, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_q_h64 = {32, 16, 16, 16, 4, 8, 4, 8};
sdpa_config_t xehpg_q_h64_s128 = {16, 16, 16, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_q_h64_s64 = {32, 8, 32, 8, 2, 8, 2, 8};
sdpa_config_t xehpg_q_h64_s32 = {8, 8, 16, 8, 4, 8, 4, 8};

sdpa_config_t xehpg_q_h64_s64_2nd = {8, 8, 8, 8, 8, 2, 8, 2};
sdpa_config_t xehpg_q_h64_s128_2nd = {16, 8, 8, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h64_2nd = {16, 16, 8, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_h128 = {16, 16, 32, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_h128_s32 = {16, 16, 16, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h128_2nd = {8, 16, 16, 8, 16, 1, 8, 2};

sdpa_config_t xehpg_q_h128 = {8, 32, 16, 32, 8, 2, 8, 2};
sdpa_config_t xehpg_q_h128_s64 = {8, 8, 16, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h128_s512 = {16, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h128_2nd = {32, 16, 16, 8, 16, 1, 8, 2};
sdpa_config_t xehpg_q_h128_s96_2nd = {8, 8, 8, 8, 16, 2, 16, 2};

sdpa_config_t xehpg_h256 = {16, 16, 32, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h256_s128 = {8, 16, 32, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_h256_s32 = {8, 16, 32, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_q_h256 = {16, 16, 64, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_q_h256_s512 = {16, 16, 32, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h256_s64 = {8, 8, 32, 8, 8, 4, 8, 4};

sdpa_config_t xehpg_h256_2nd = {8, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s64_2nd = {16, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s32_2nd = {16, 16, 32, 8, 16, 1, 8, 2};

sdpa_config_t xehpg_q_h256_2nd = {32, 8, 32, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h256_s96_2nd = {8, 8, 16, 8, 16, 2, 16, 2};

sdpa_config_t xehpg_q_h512_s64 = {8, 8, 64, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h512_s128 = {8, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xehpg_q_h512_s256 = {16, 8, 64, 8, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h512 = {8, 16, 64, 8, 16, 2, 8, 4};

sdpa_config_t xehpg_q_h512_s64_2nd = {8, 16, 32, 8, 32, 1, 16, 2};
sdpa_config_t xehpg_q_h512_s256_2nd = {16, 8, 32, 8, 16, 2, 16, 2};
sdpa_config_t xehpg_q_h512_2nd = {16, 8, 16, 8, 32, 1, 32, 1};

sdpa_config_t xehpg_h512 = {8, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xehpg_h512_2nd = {8, 8, 32, 8, 16, 1, 16, 1};

sdpa_config_t xehpc_h32 = {16, 64, 32, 16, 4, 2, 1, 8};
sdpa_config_t xehpc_h32_s32 = {16, 16, 16, 16, 2, 4, 2, 4};
sdpa_config_t xehpc_h32_2nd = {16, 64, 16, 16, 8, 1, 2, 4};

sdpa_config_t xehpc_h64 = {16, 64, 32, 16, 8, 2, 2, 8};
sdpa_config_t xehpc_h64_s64 = {32, 32, 32, 16, 4, 2, 2, 4};
sdpa_config_t xehpc_h64_s32 = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xehpc_h64_2nd = {32, 32, 32, 16, 4, 1, 2, 2};
sdpa_config_t xehpc_h64_s64_2nd = {16, 16, 16, 16, 4, 1, 4, 1};

sdpa_config_t xehpc_q_h64_s64 = {16, 16, 16, 16, 4, 4, 4, 4};
sdpa_config_t xehpc_q_h64_s384 = {16, 64, 16, 32, 8, 2, 4, 4};
sdpa_config_t xehpc_q_h64_s1024 = {16, 64, 16, 16, 16, 1, 4, 4};
sdpa_config_t xehpc_q_h64 = {16, 64, 16, 32, 8, 1, 4, 2};

sdpa_config_t xehpc_q_h64_s96_2nd = {16, 16, 16, 16, 8, 1, 4, 1};
sdpa_config_t xehpc_q_h64_s256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h64_s1152_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h64_2nd = {64, 16, 16, 16, 16, 2, 16, 2};

sdpa_config_t xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8};
sdpa_config_t xehpc_h128_s64 = {16, 32, 32, 32, 4, 2, 4, 2};
sdpa_config_t xehpc_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h128_2nd = {32, 32, 32, 16, 8, 1, 4, 2};

sdpa_config_t xehpc_q_h128 = {16, 64, 16, 32, 16, 1, 8, 2};
sdpa_config_t xehpc_q_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_q_h128_s128 = {16, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpc_q_h128_s128_integrated = {16, 16, 16, 16, 8, 2, 8, 2};

sdpa_config_t xehpc_q_h128_2nd = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h128_2nd_integrated = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_q_h128_s96_2nd = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_q_h128_s512_2nd = {16, 16, 16, 16, 16, 2, 8, 2};

sdpa_config_t xehpc_h256 = {16, 32, 32, 32, 8, 4, 8, 4};
sdpa_config_t xehpc_h256_s64 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xehpc_h256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t xehpc_h512 = {32, 16, 64, 16, 8, 4, 8, 4};
sdpa_config_t xehpc_h512_s64 = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h512_s128_2nd = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_h512_s512_2nd = {32, 16, 64, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_h512_s1024_2nd = {64, 16, 32, 16, 16, 2, 16, 2};
sdpa_config_t xehpc_h512_2nd = {32, 16, 64, 16, 16, 1, 16, 1};

sdpa_config_t xehpc_h512_integrated = {16, 16, 32, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_h512_s128_integrated = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h512_s256_2nd_integrated = {16, 16, 64, 16, 8, 1, 8, 1};
sdpa_config_t xehpc_h512_s1024_2nd_integrated = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h512_2nd_integrated = {16, 16, 64, 16, 16, 2, 16, 2};

sdpa_config_t xehpc_q_h512_s64_2nd_integrated = {16, 32, 64, 32, 16, 2, 8, 2};
sdpa_config_t xehpc_q_h512_s128_2nd_integrated = {16, 16, 64, 16, 8, 1, 32, 1};
sdpa_config_t xehpc_q_h512_s256_2nd_integrated = {16, 32, 64, 32, 16, 2, 8, 2};
sdpa_config_t xehpc_q_h512_s512_2nd_integrated = {16, 16, 64, 16, 4, 4, 8, 4};
sdpa_config_t xehpc_q_h512_s1024_2nd_integrated = {16, 16, 64, 16, 16, 1, 16, 1};
sdpa_config_t xehpc_q_h512_2nd_integrated = {32, 16, 64, 16, 8, 1, 16, 1};

sdpa_config_t xehpc_q_h512_integrated = {16, 32, 32, 32, 16, 1, 16, 1};

sdpa_config_t xehpc_q_h512 = {16, 32, 64, 16, 16, 2, 8, 4};
sdpa_config_t xehpc_q_h512_s128 = {16, 16, 64, 16, 8, 2, 8, 2};

sdpa_config_t xehpc_q_h512_s512_2nd = {16, 16, 64, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_q_h512_s1024_2nd = {64, 16, 64, 16, 16, 2, 16, 2};
sdpa_config_t xehpc_q_h512_2nd = {16, 16, 64, 16, 16, 2, 16, 2};

sdpa_config_t xe2_q_h64 = {16, 64, 16, 32, 16, 1, 8, 2};
sdpa_config_t xe2_q_h64_s1024_integrated = {16, 64, 16, 32, 8, 4, 4, 8};
sdpa_config_t xe2_q_h64_s512 = {16, 64, 16, 32, 8, 4, 4, 8};
sdpa_config_t xe2_q_h64_s384 = {16, 64, 16, 16, 16, 1, 4, 4};
sdpa_config_t xe2_q_h64_s128 = {16, 64, 16, 32, 8, 1, 4, 2};
sdpa_config_t xe2_q_h64_s128_integrated = {16, 16, 16, 16, 4, 4, 4, 4};
sdpa_config_t xe2_q_h64_s32 = {16, 16, 16, 16, 4, 4, 4, 4};

sdpa_config_t xe2_q_h64_2nd = {16, 16, 16, 16, 16, 1, 8, 1};
sdpa_config_t xe2_q_h64_2nd_integrated = {16, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xe2_q_h64_s96_2nd_integrated = {16, 16, 16, 16, 8, 1, 4, 1};
sdpa_config_t xe2_q_h64_s384_2nd_integrated = {64, 16, 16, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h64_s64_2nd = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xe2_q_h64_s128_2nd = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xe2_q_h64_s384_2nd = {16, 16, 16, 16, 16, 1, 4, 1};
sdpa_config_t xe2_q_h64_s512_2nd = {64, 16, 16, 16, 8, 1, 8, 1};
sdpa_config_t xe2_q_h64_s768_2nd = {64, 16, 16, 16, 16, 1, 8, 1};

sdpa_config_t xe2_q_h256 = {16, 64, 16, 32, 32, 1, 16, 2};
sdpa_config_t xe2_q_h256_s384 = {16, 32, 32, 32, 8, 2, 8, 2};
sdpa_config_t xe2_q_h256_s128 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xe2_q_h256_s128_integrated = {16, 32, 32, 32, 8, 2, 8, 2};
sdpa_config_t xe2_q_h256_s64_integrated = {16, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xe2_q_h256_s64 = {16, 32, 64, 16, 8, 2, 4, 4};

sdpa_config_t xe2_q_h256_2nd_integrated = {32, 16, 64, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h256_s1152_2nd_integrated = {16, 16, 64, 16, 4, 1, 4, 1};
sdpa_config_t xe2_q_h256_s768_2nd_integrated = {64, 16, 16, 16, 16, 1, 16, 1};
sdpa_config_t xe2_q_h256_s512_2nd_integrated = {32, 32, 32, 16, 16, 1, 8, 2};
sdpa_config_t xe2_q_h256_s384_2nd_integrated = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t* choose_config_xehpg(int head_size, int seq, bool thin_q, bool quantized, bool is_pa) {
    if (head_size <= 32) {
        if (seq <= 0 && is_pa)
            return &xehpg_h32;
        if (quantized && seq >= 128) {
            if (thin_q)
                return &xehpg_q_h32_2nd;
            return &xehpg_q_h32;
        }
        if (thin_q)
            return &xehpg_h32_2nd;
        if (seq <= 32)
            return &xehpg_h32_s32;
        if (seq <= 64)
            return &xehpg_h32_s64;
        if (seq <= 256)
            return &xehpg_h32_s256;
        return &xehpg_h32;
    } else if (head_size <= 64) {
        if (seq <= 0 && is_pa)
            return &xehpg_h64;
        if (quantized) {
            if (thin_q) {
                if (seq <= 64)
                    return &xehpg_q_h64_s64_2nd;
                if (seq <= 128)
                    return &xehpg_q_h64_s128_2nd;
                return &xehpg_q_h64_2nd;
            } else {
                if (seq <= 32)
                    return &xehpg_q_h64_s32;
                if (seq <= 64)
                    return &xehpg_q_h64_s64;
                if (seq <= 128)
                    return &xehpg_q_h64_s128;
                return &xehpg_q_h64;
            }
        }
        if (thin_q)
            return &xehpg_h64_2nd;
        if (seq <= 64)
            return &xehpg_h64_s64;
        if (seq <= 128)
            return &xehpg_h64_s128;
        return &xehpg_h64;
    } else if (head_size <= 128) {
        if (seq <= 0 && is_pa)
            return &xehpg_h128;
        if (quantized) {
            if (thin_q) {
                if (seq <= 1)
                    return &xehpg_q_h128_2nd;
                if (seq <= 96)
                    return &xehpg_q_h128_s96_2nd;
                return &xehpg_q_h128_2nd;
            }
            if (seq <= 64)
                return &xehpg_q_h128_s64;
            if (seq <= 512)
                return &xehpg_q_h128_s512;
            return &xehpg_q_h128;
        }
        if (thin_q) {
            if (seq <= 256)
                return &xehpg_q_h128_2nd;
            return &xehpg_h128_2nd;
        }
        if (seq <= 32)
            return &xehpg_h128_s32;
        return &xehpg_h128;
    } else if (head_size <= 256) {
        if (seq <= 0 && is_pa)
            return &xehpg_h256;
        if (thin_q) {
            if (quantized) {
                if (seq <= 96)
                    return &xehpg_q_h256_s96_2nd;
                return &xehpg_q_h256_2nd;
            }
            if (seq <= 32)
                return &xehpg_h256_s32_2nd;
            if (seq <= 64)
                return &xehpg_h256_s64_2nd;
            return &xehpg_h256_2nd;
        }
        if (quantized) {
            if (seq <= 64)
                return &xehpg_q_h256_s64;
            if (seq <= 512)
                return &xehpg_q_h256_s512;
            return &xehpg_q_h256;
        }
        if (seq <= 32)
            return &xehpg_h256_s32;
        if (seq <= 128)
            return &xehpg_h256_s128;
        return &xehpg_h256;
    } else if (head_size <= 512) {
        if (seq <= 0 && is_pa)
            return &xehpg_h512;
        if (quantized) {
            if (thin_q) {
                if (seq <= 64)
                    return &xehpg_q_h512_s64_2nd;
                if (seq <= 256)
                    return &xehpg_q_h512_s256_2nd;
                return &xehpg_q_h512_2nd;
            }
            if (seq <= 64)
                return &xehpg_q_h512_s64;
            if (seq <= 128)
                return &xehpg_q_h512_s128;
            if (seq <= 256)
                return &xehpg_q_h512_s256;
            return &xehpg_q_h512;
        }
        if (thin_q) {
            return &xehpg_h512_2nd;
        }
        return &xehpg_h512;
    }
    return nullptr;
}

sdpa_config_t* choose_config_xehpc(int head_size, int seq, bool thin_q, bool quantized, bool is_integrated, bool is_pa) {
    if (head_size <= 32) {
        if (seq <= 0 && is_pa)
            return &xehpc_h32;
        if (thin_q)
            return &xehpc_h32_2nd;
        if (seq <= 32)
            return &xehpc_h32_s32;
        return &xehpc_h32;
    } else if (head_size <= 64) {
        if (seq <= 0 && is_pa)
            return &xehpc_h64;
        if (thin_q) {
            if (quantized) {
                if (seq <= 96)
                    return &xehpc_q_h64_s96_2nd;
                if (seq <= 256)
                    return &xehpc_q_h64_s256_2nd;
                if (seq <= 1152)
                    return &xehpc_q_h64_s1152_2nd;
                return &xehpc_q_h64_2nd;
            }

            if (seq <= 64)
                return &xehpc_h64_s64_2nd;
            return &xehpc_h64_2nd;
        }
        if (quantized) {
            if (seq <= 64)
                return &xehpc_q_h64_s64;
            if (seq <= 384)
                return &xehpc_q_h64_s384;
            if (seq <= 1024)
                return &xehpc_q_h64_s1024;
            return &xehpc_q_h64;
        }
        if (seq <= 32)
            return &xehpc_h64_s32;
        if (seq <= 64)
            return &xehpc_h64_s64;
        return &xehpc_h64;
    } else if (head_size <= 128) {
        if (seq <= 0 && is_pa)
            return &xehpc_h128;
        if (quantized) {
            if (thin_q) {
                if (is_integrated) {
                    return &xehpc_q_h128_2nd_integrated;
                }
                if (seq <= 96)
                    return &xehpc_q_h128_s96_2nd;
                if (seq <= 512)
                    return &xehpc_q_h128_s512_2nd;
                return &xehpc_q_h128_2nd;
            }
            if (is_integrated) {
                if (seq <= 128) {
                    return &xehpc_q_h128_s128_integrated;
                }
            }
            if (seq <= 32)
                return &xehpc_q_h128_s32;
            if (seq <= 128)
                return &xehpc_q_h128_s128;
            return &xehpc_q_h128;
        }
        if (is_integrated)
            return &xehpc_q_h128_2nd_integrated;
        if (thin_q)
            return &xehpc_h128_2nd;
        if (seq <= 32)
            return &xehpc_h128_s32;
        if (seq <= 64)
            return &xehpc_h128_s64;
        return &xehpc_h128;
    } else if (head_size <= 256) {
        if (seq <= 0 && is_pa)
            return &xehpc_h256;
        if (thin_q)
            return &xehpc_h256_2nd;
        if (seq <= 64)
            return &xehpc_h256_s64;
        return &xehpc_h256;
    } else if (head_size <= 512) {
        if (seq <= 0 && is_pa)
            return &xehpc_h512;
        if (thin_q) {
            if (quantized) {
                if (is_integrated) {
                    if (seq <= 64)
                        return &xehpc_q_h512_s64_2nd_integrated;
                    if (seq <= 128)
                        return &xehpc_q_h512_s128_2nd_integrated;
                    if (seq <= 256)
                        return &xehpc_q_h512_s256_2nd_integrated;
                    if (seq <= 512)
                        return &xehpc_q_h512_s512_2nd_integrated;
                    if (seq <= 1024)
                        return &xehpc_q_h512_s1024_2nd_integrated;
                    return &xehpc_q_h512_2nd_integrated;
                }
                if (seq <= 512)
                    return &xehpc_q_h512_s512_2nd;
                if (seq <= 1024)
                    return &xehpc_q_h512_s1024_2nd;
                return &xehpc_q_h512_2nd;
            }

            if (is_integrated) {
                if (seq <= 256)
                    return &xehpc_h512_s256_2nd_integrated;
                if (seq <= 1024)
                    return &xehpc_h512_s1024_2nd_integrated;
                return &xehpc_h512_2nd_integrated;
            }
            if (seq <= 128)
                return &xehpc_h512_s128_2nd;
            if (seq <= 512)
                return &xehpc_h512_s512_2nd;
            if (seq <= 1024)
                return &xehpc_h512_s1024_2nd;
            return &xehpc_h512_2nd;
        }

        if (quantized) {
            if (is_integrated)
                return &xehpc_q_h512_integrated;
            if (seq <= 128)
                return &xehpc_q_h512_s128;
            return &xehpc_q_h512;
        }
        if (is_integrated) {
            if (seq <= 128)
                return &xehpc_h512_s128_integrated;
            return &xehpc_h512_integrated;
        }
        if (seq <= 64)
            return &xehpc_h512_s64;
        return &xehpc_h512;
    }
    return nullptr;
}

sdpa_config_t* choose_config_xe2(int head_size, int seq, bool thin_q, bool quantized, bool is_integrated, bool is_pa) {
    if (head_size <= 64) {
        if (quantized) {
            if (thin_q) {
                if (is_integrated) {
                    if (seq <= 96)
                        return &xe2_q_h64_s96_2nd_integrated;
                    if (seq <= 384)
                        return &xe2_q_h64_s384_2nd_integrated;
                    return &xe2_q_h64_2nd_integrated;
                }
                if (seq <= 64)
                    return &xe2_q_h64_s64_2nd;
                if (seq <= 128)
                    return &xe2_q_h64_s128_2nd;
                if (seq <= 384)
                    return &xe2_q_h64_s384_2nd;
                if (seq <= 512)
                    return &xe2_q_h64_s512_2nd;
                if (seq <= 768)
                    return &xe2_q_h64_s768_2nd;
                return &xe2_q_h64_2nd;
            }
            if (seq <= 32)
                return &xe2_q_h64_s32;
            if (is_integrated) {
                if (seq <= 128)
                    return &xe2_q_h64_s128_integrated;
            }
            if (seq <= 128)
                return &xe2_q_h64_s128;
            if (seq <= 384)
                return &xe2_q_h64_s384;
            if (seq <= 512)
                return &xe2_q_h64_s512;
            if (is_integrated) {
                if (seq <= 1024)
                    return &xe2_q_h64_s1024_integrated;
            }
            return &xe2_q_h64;
        }
    }

    if (head_size <= 128) {
        return choose_config_xehpc(head_size, seq, thin_q, quantized, is_integrated, is_pa);
    }

    if (head_size <= 256) {
        if (quantized) {
            if (is_integrated) {
                if (thin_q) {
                    if (seq < 384)
                        return &xe2_q_h256_s384_2nd_integrated;
                    if (seq < 512)
                        return &xe2_q_h256_s512_2nd_integrated;
                    if (seq < 768)
                        return &xe2_q_h256_s768_2nd_integrated;
                    if (seq < 1152)
                        return &xe2_q_h256_s1152_2nd_integrated;
                    return &xe2_q_h256_2nd_integrated;
                }
                if (seq <= 64)
                    return &xe2_q_h256_s64_integrated;
                if (seq <= 128)
                    return &xe2_q_h256_s128_integrated;
            }
            if (!thin_q) {
                if (seq <= 64)
                    return &xe2_q_h256_s64;
                if (seq <= 128)
                    return &xe2_q_h256_s128;
                if (seq <= 384)
                    return &xe2_q_h256_s384;
                return &xe2_q_h256;
            }
        }
    }
    return choose_config_xehpc(head_size, seq, thin_q, quantized, is_integrated, is_pa);
}

}  // namespace

std::string SDPAMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";

    return base_options + extra_options;
}

void SDPAMicroGenerator::init_sdpa_configuration(const kernel_impl_params& impl_param, sdpa_configuration& sdpa_config) {
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

KernelData SDPAMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    std::vector<micro::Package> gemms(2);  // KQ and VS
    sdpa_configuration sdpa_config;
    init_sdpa_configuration(params, sdpa_config);
    init_microkernels(params, sdpa_config, gemms[kq_id], gemms[vs_id], m_is_prefill);

    const auto& device_info = params.get_device_info();
    auto jit = get_jit_constants(params, gemms[kq_id], gemms[vs_id]);

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = kernel_language::OCLC_V2;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = false;
    kd.code->has_microkernels = true;
    kd.code->str = build_code(get_kernel_name(), jit, kd.code->entry_point);

    kd.params.arguments = get_arguments_desc(params);
    kd.update_dispatch_data_func = get_dispatch_data_func();

    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    /* Generate microkernel shims */
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = static_cast<int32_t>(get_subgroup_size(device_info.arch));
    shim_options.useTileOps = true;
    shim_options.decorator = "kq";

    kd.code->jit += generateShim(gemms[kq_id], micro::HostLanguage::OpenCL_C, shim_options);

    shim_options.microkernelID++;
    shim_options.decorator = "vs";
    kd.code->jit += generateShim(gemms[vs_id], micro::HostLanguage::OpenCL_C, shim_options);

    if (gemms[kq_id].grfMin > 128 || gemms[vs_id].grfMin > 128) {
        kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    for (auto& p : gemms) {
        kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(p));
    }

    return kd;
}

// Use 'maybe_unused' to avoid DPC++ build error
[[maybe_unused]] const bool kq_common_scales = false;
[[maybe_unused]] const bool kq_common_zp = false;
[[maybe_unused]] const bool vs_common_scales = false;
[[maybe_unused]] const bool vs_common_zp = false;

JitConstants SDPAMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const {
    auto jit = make_base_jit_constants(params);
    sdpa_configuration config;
    init_sdpa_configuration(params, config);

    if (config.is_paged_attention) {
        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(paged_attention::PagedAttentionInputIdx::ALIBI).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = desc->has_scores_output();

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;
        constexpr static std::array input_ids = {paged_attention::PagedAttentionInputIdx::QUERY,
                                                 paged_attention::PagedAttentionInputIdx::KEY,
                                                 paged_attention::PagedAttentionInputIdx::VALUE,
                                                 paged_attention::PagedAttentionInputIdx::SUBSEQUENCE_BEGINS};
        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids[i];
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_scale_input) {
            const size_t tensor_id = paged_attention::PagedAttentionInputIdx::SCALE;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(4), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_alibi) {
            const size_t tensor_id = paged_attention::PagedAttentionInputIdx::ALIBI;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(5), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
        if (has_scores_output) {
            jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(1), params.output_layouts[1], out_offsets_map.at(1)));
        }
    } else {
        const auto desc = params.typed_desc<scaled_dot_product_attention>();
        jit.add(make_tensors_jit_constants(params));
        if (desc->has_sink_input) {
            auto sink_layout = params.input_layouts[ScaledDotProductAttentionInputIdx::SINK];
            std::cout << "sink_layout : " << sink_layout.to_short_string() << std::endl;
            if (sink_layout.count() != micro_get_head_size(params, 0))
                OPENVINO_THROW("Currently only supporting per-head sink.Sink_layout : ",
                                sink_layout.to_short_string(), " heads_num  :", micro_get_head_size(params, 0));
            jit.make("SINK_DATA_T", to_ocl_type(sink_layout.data_type));
            jit.make("HAS_SINK_INPUT", 1);
        }
    }
    const auto& device_info = params.get_device_info();

    const auto& Q = params.input_layouts[0];
    const auto& K = params.input_layouts[1];
    const auto& V = params.input_layouts[2];
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

    jit.make("D_MAX", d_max);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));
    jit.make("INVERT_SCALE", false);
    jit.make("SCALE_DATA_T", "half");
    jit.make("HEAD_SIZE", k_head_size);

    auto data_inputs_num = micro_get_input_num(params, config);

    size_t attn_input_idx = 3;
    size_t scale_input_idx = 4;
    jit.make("IS_CAUSAL", config.is_causal);
    if (!config.is_paged_attention) {
        if (config.has_const_attn_mask_val) {
            jit.make("WITH_ATTN_MASK", 0);
            jit.make("STATIC_SCALAR_ATTN_MASK_VALUE", config.attn_mask_val);
            // scale_input_idx -= 1;
        } else {
            jit.make("WITH_ATTN_MASK", data_inputs_num > attn_input_idx);
        }
    } else {
        jit.make("WITH_ATTN_MASK", 0);
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

    jit.make("TRANSPOSE_K", false);
    jit.make("IS_PAGED_ATTENTION", config.is_paged_attention ? 1 : 0);
    jit.make("KV_HEADS_NUM", config.kv_heads_num);
    jit.make("HEADS_NUM", config.heads_num);

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
    if (config.is_kv_compressed) {
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

    jit.make("KEY_ELEMENTS_PER_BYTE", elems_per_byte(params.input_layouts[1].data_type));
    jit.make("VAL_ELEMENTS_PER_BYTE", elems_per_byte(params.input_layouts[2].data_type));

    int tile_k = gemm_kq.getSetting("wg_tile_m");
    int tile_q = gemm_kq.getSetting("wg_tile_n");
    int tile_v = gemm_vs.getSetting("wg_tile_m");

    const ov::Dimension n_keys = micro_get_seq_length(params, 1);
    const ov::Dimension n_queries = micro_get_seq_length(params, 0);
    // const ov::Dimension n_values = micro_get_seq_length(params, 2);
    const ov::Dimension n_values = ov::Dimension(v_head_size);

    bool d_full = (head_size == static_cast<size_t>(d_max));
    bool v_full = (head_size == static_cast<size_t>(tile_v));
    bool k_full = !n_keys.is_dynamic() && (n_keys.get_length() % tile_k) == 0;
    bool q_full = !n_queries.is_dynamic() && (n_queries.get_length() % tile_q) == 0;

    auto Q_num_heads_dim = micro_get_num_heads(params, 0);
    auto K_num_heads_dim = micro_get_num_heads(params, 1);

    jit.make("REMAINDER_K", !k_full);
    jit.make("KV_GROUP_SIZE", Q_num_heads_dim / K_num_heads_dim);

    if (d_full) {
        if (ldq % 4 == 0)
            jit.make("BLOCK_Q", 1);
        // TODO: Causes accuracy drop for static SD model. Enable back once the issue is resolved
        // if (lda % 4 == 0 && v_full)
        //     jit.make("BLOCK_A", 1);
        jit.make("REMAINDER_Q", !q_full);
    } else if (device_info.arch >= gpu_arch::xe_hpc) {
        auto vbytes = n_values.get_length() * ov::element::Type(V.data_type).size();
        if (lda % 16 == 0 && vbytes % 4 == 0)
            jit.make("BLOCK_2D_A", 1);
    }

    if (device_info.arch >= gpu_arch::xe_hpc) {
        jit.make("PREFETCH_MASK", 1);
        jit.make("PREFETCH_K0", 1);
        jit.make("PREFETCH_K", 1);
        jit.make("PREFETCH_V", 1);
        bool no_rem = d_full && v_full && k_full;
        jit.make("PREFETCH_REMAINDER", !no_rem);
        jit.make("PREFETCH_D_MAX", std::min<int64_t>(d_max, 64));
    }

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

    if (data_inputs_num > 3 && !config.has_const_attn_mask_val) {
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

Arguments SDPAMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    sdpa_configuration config;
    init_sdpa_configuration(params, config);
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    auto data_inputs_num = micro_get_input_num(params, config);

    args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::KEY});   // K
    args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::QUERY});   // Q
    args.push_back({ArgumentDescriptor::Types::INPUT, ScaledDotProductAttentionInputIdx::VALUE});   // V
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // A

    if (config.is_paged_attention) {
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        if (!config.has_const_scale_val)
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});   // scale
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // blocked_indexes_start_and_gws_mapping
    } else {
        const uint32_t attn_mask_idx = ScaledDotProductAttentionInputIdx::ATTN_MASK;
        if (config.input_num > attn_mask_idx && !config.has_const_attn_mask_val)
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

DispatchDataFunc SDPAMicroGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(3);

        auto params = impl_param;
        if (!params.is_dynamic()) {
            const auto& device_info = params.get_device_info();
            const auto& gemms = kd.micro_kernels;
            const auto& gemm_kq = gemms[kq_id]->p;

            const auto& out = params.output_layouts[0];
            const auto& out_ps = out.get_partial_shape();

            const auto v_head_size = micro_get_head_size(params, 2);
            const auto head_num = micro_get_num_heads(params, 0);

            auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
            auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m") * gemm_kq.getSetting("sg_per_wg_n");

            const ov::Dimension n_keys = micro_get_aligned_seq_length(params, 1, wg_tile_q);
            const ov::Dimension n_queries = micro_get_aligned_seq_length(params, 0, wg_tile_q);

            wgs.local = {get_subgroup_size(device_info.arch), (size_t)sg_per_wg, 1};
            wgs.global = wgs.local;

            wgs.global[0] *= ceil_div(n_queries.get_length(), wg_tile_q);
            if (params.is_type<paged_attention>()) {
                wgs.global[1] *= head_num;
                wgs.global[2] *= 1;
            } else {
                wgs.global[1] *= out_ps[1].get_length();
                wgs.global[2] *= out_ps[0].get_length();
            }

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
        }
    }};
}

size_t SDPAMicroGenerator::get_tile_qsize(const KernelData& kernel_data) {
    OPENVINO_ASSERT(kernel_data.micro_kernels.size() > 0, "[GPU] Invalid kernels passed to get_tile_qsize() function");

    const auto& gemms = kernel_data.micro_kernels;
    const auto wg_tile_q = gemms[kq_id]->p.getSetting("wg_tile_n");
    return wg_tile_q;
}

std::mutex SDPAMicroGenerator::m;
void SDPAMicroGenerator::init_microkernels(const kernel_impl_params& params,
                                           const sdpa_configuration& configuration,
                                           micro::Package& gemm_kq,
                                           micro::Package& gemm_vs,
                                           bool is_prefill) {
    // TODO: Remove once micro API is thread safe
    std::lock_guard<std::mutex> l(m);
    const auto& Q = params.input_layouts[0];
    const auto& K = params.input_layouts[1];
    const auto& V = params.input_layouts[2];
    auto& out = params.output_layouts[0];
    const auto& out_ps = out.get_partial_shape();
    const auto& device_info = params.get_device_info();

    const auto k_head_size = micro_get_head_size(params, 1);
    const auto v_head_size = micro_get_head_size(params, 2);
    const auto d_max = get_d_max(k_head_size);

    const ov::Dimension n_keys = micro_get_seq_length(params, 1);
    const ov::Dimension n_queries = micro_get_seq_length(params, 0);
    const ov::Dimension n_values = ov::Dimension(v_head_size);
    const auto batch = out_ps[0] * out_ps[1];

    GPU_DEBUG_TRACE_DETAIL << "\nconfiguration.is_kv_compressed = " << configuration.is_kv_compressed << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "k_head_size = " << k_head_size << ", v_head_size = " << v_head_size << ", d_max = " << d_max << ", batch = " << batch << "\n";
    GPU_DEBUG_TRACE_DETAIL << "n_keys = " << n_keys.to_string() << ", n_queries = " << n_queries.to_string() << ", n_values = " << n_values.to_string() << "\n";

    /* Retrieve pre-tuned kernel configuration */
    sdpa_config_t* config = nullptr;
    bool thin_q = (!n_queries.is_dynamic() && n_queries.get_length() <= 16) || !is_prefill;
    bool is_integrated = device_info.dev_type == device_type::integrated_gpu;

    bool is_quantized =
        (K.data_type == ov::element::u8 || K.data_type == ov::element::i8) || (V.data_type == ov::element::u8 || V.data_type == ov::element::i8);
    int32_t nkeys_v = n_keys.is_dynamic() ? 0 : n_keys.get_length();

    bool is_paged_attention = false;
    if (params.is_type<cldnn::paged_attention>()) {
        is_paged_attention = true;
    }

    GPU_DEBUG_TRACE_DETAIL << "k_head_size = " << k_head_size << ", nkeys_v = " << nkeys_v << "\n";
    GPU_DEBUG_TRACE_DETAIL << "thin_q = " << thin_q << ", is_quantized = " << is_quantized << "\n";
    switch (device_info.arch) {
    case gpu_arch::xe_hpg: {
        config = choose_config_xehpg(static_cast<int32_t>(k_head_size), static_cast<int32_t>(nkeys_v), thin_q, is_quantized, is_paged_attention);
        break;
    }
    case gpu_arch::xe_hpc:
        config = choose_config_xehpc(static_cast<int32_t>(k_head_size), static_cast<int32_t>(nkeys_v), thin_q, is_quantized, is_integrated, is_paged_attention);
        break;
    case gpu_arch::xe2:
    case gpu_arch::xe3: {
        config = choose_config_xe2(static_cast<int32_t>(k_head_size), static_cast<int32_t>(nkeys_v), thin_q, is_quantized, is_integrated, is_paged_attention);
        break;
    }
    default:
        break;
    }

    OPENVINO_ASSERT(config != nullptr);

    /* Get device information */
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    micro::GEMMProblem problem;
    problem.Ta_ext = convert_type(K.data_type);
    problem.Tb_ext = convert_type(Q.data_type);

    problem.Ta = problem.Tb = micro::Type::f16;
    problem.Tc = problem.Tc_ext = micro::Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.A.layout = micro::MatrixLayout::T;

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;

    const bool use_asymmetric_quantization = configuration.use_asymmetric_quantization;
    const auto key_cache_id = micro_get_key_cache_id(params);
    if (configuration.is_kv_compressed && !kq_common_scales) {
        const auto& key_cache_comp_scale = params.input_layouts[key_cache_id];
        const auto scale_dt = convert_type(key_cache_comp_scale.data_type);
        problem_kq.Ta_scale = scale_dt;
        problem_kq.A_scale.setAlignment(scale_dt.size());
        problem_kq.A_scale.layout = micro::MatrixLayout::N;
        problem_kq.asPtrDims = 2;
        GPU_DEBUG_TRACE_DETAIL << "kq: key_cache_id = " << key_cache_id << std::endl;
    }

    if (configuration.is_kv_compressed && use_asymmetric_quantization) {
        const auto& key_cache_comp_zp = params.input_layouts[key_cache_id + 2];
        const auto zp_dt = convert_type(key_cache_comp_zp.data_type);
        problem_kq.Tao = zp_dt;
        problem_kq.AO.setAlignment(zp_dt.size());
        problem_kq.AO.layout = micro::MatrixLayout::N;
        problem_kq.aoPtrDims = kq_common_zp ? 0 : 2;
        problem_kq.aOffset = micro::ABOffset::Calc;
    }

    if (configuration.is_kv_compressed) {
        problem_kq.aqGroupM = 1;
        problem_kq.aqGroupK = (kq_common_scales || kq_common_zp) ? 1 : static_cast<int>(k_head_size);
    }

    opts_kq.scaleA = configuration.is_kv_compressed && !kq_common_scales;
    opts_kq.offsetA = configuration.is_kv_compressed && use_asymmetric_quantization;

    problem_kq.B.layout = micro::MatrixLayout::Pr;
    problem_kq.C.layout = micro::MatrixLayout::T;
    problem_kq.A.setAlignment(micro::alignment_for_ld(k_head_size * problem.Ta));
    problem_kq.B.setAlignment(64);  // Q is packed in VNNI format in SLM
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = static_cast<uint16_t>(d_max);
    problem_kq.B.tileC = static_cast<uint16_t>(get_subgroup_size(device_info.arch));

    /* Set up problem size information */
    micro::SizeParams sizes;
    sizes.m = n_keys.is_dynamic() ? 0 : n_keys.get_length();
    sizes.n = n_queries.is_dynamic() ? 0 : n_queries.get_length();
    sizes.k = static_cast<int64_t>(k_head_size);
    sizes.batch = batch.is_dynamic() ? 0 : batch.get_length();

    GPU_DEBUG_TRACE_DETAIL << "kq: sizes = {" << sizes.m << ", " << sizes.n << ", " << sizes.k << ", " << sizes.batch << "}\n";
    GPU_DEBUG_TRACE_DETAIL << "config->wg_m_kq = " << config->wg_m_kq << ", config->wg_n_kq = " << config->wg_n_kq << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "config->unroll_m_kq = " << config->unroll_m_kq << ", config->unroll_n_kq = " << config->unroll_n_kq << std::endl;

    /* Set up microkernel requirements */
    std::vector<micro::StrategyRequirement> reqs_kq;
    reqs_kq.push_back(micro::StrategyRequirement::UnrollM == config->unroll_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::UnrollN == config->unroll_n_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGM == config->wg_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGN == config->wg_n_kq);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq = micro::select_gemm_microkernel(opts_kq, hw_info, sizes, problem_kq, reqs_kq);
    } catch (const std::runtime_error& ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create KQ sdpa_micro kernel: " << ex.what() << "\n";
        throw;
    }

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;

    /* Update for second GEMM: V*S */
    auto problem_vs = problem;
    problem_vs.Ta_ext = convert_type(V.data_type);
    problem_vs.A.layout = micro::MatrixLayout::N;

    const auto value_cache_id = micro_get_value_cache_id(params);
    if (configuration.is_kv_compressed && !vs_common_scales) {
        const auto& value_cache_comp_scale = params.input_layouts[value_cache_id];
        auto scale_dt = convert_type(value_cache_comp_scale.data_type);
        problem_vs.Ta_scale = scale_dt;
        problem_vs.A_scale.setAlignment(scale_dt.size());
        problem_vs.A_scale.layout = micro::MatrixLayout::N;
        problem_vs.asPtrDims = 2;
        GPU_DEBUG_TRACE_DETAIL << "vs: value_cache_id = " << value_cache_id << std::endl;
    }

    if (configuration.is_kv_compressed && use_asymmetric_quantization) {
        const auto& value_cache_comp_zp = params.input_layouts[value_cache_id + 2];
        auto zp_dt = convert_type(value_cache_comp_zp.data_type);
        problem_vs.Tao = zp_dt;
        problem_vs.AO.setAlignment(zp_dt.size());
        problem_vs.AO.layout = micro::MatrixLayout::N;
        problem_vs.aoPtrDims = vs_common_zp ? 0 : 2;
        problem_vs.aOffset = micro::ABOffset::Calc;
    }

    if (configuration.is_kv_compressed) {
        problem_vs.aqGroupM = (vs_common_scales || vs_common_zp) ? 1 : static_cast<int>(micro::rnd_up_pow2(v_head_size));
        problem_vs.aqGroupK = 1;
    }

    opts_vs.scaleA = configuration.is_kv_compressed && !vs_common_scales;
    opts_vs.offsetA = configuration.is_kv_compressed && use_asymmetric_quantization;

    problem_vs.B.layout = micro::MatrixLayout::Pr;
    problem_vs.C.layout = micro::MatrixLayout::N;
    problem_vs.A.setAlignment(micro::alignment_for_ld(v_head_size * problem.Ta));
    problem_vs.B.setAlignment(64);  // S is packed in SLM
    problem_vs.B.crosspack = 16;
    sizes.m = n_values.is_dynamic() ? -1 : n_values.get_length();
    sizes.n = gemm_kq.getSetting("wg_tile_n");
    sizes.k = gemm_kq.getSetting("wg_tile_m");

    GPU_DEBUG_TRACE_DETAIL << "vs: sizes = {" << sizes.m << ", " << sizes.n << ", " << sizes.k << ", " << sizes.batch << "}\n";

    /* Set up special kernel requirements */
    std::vector<micro::StrategyRequirement> reqs_vs;
    reqs_vs.push_back(micro::StrategyRequirement::UnrollM == config->unroll_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::UnrollN == config->unroll_n_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGM == config->wg_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGN == config->wg_n_vs);

    auto adjust_vs = [](micro::GEMMStrategy& strategy) {
        /* Enable dpasw */
        strategy.dpasw |= strategy.fused;
    };
    /* Ask microkernel provider for microkernel */
    try {
        gemm_vs = micro::select_gemm_microkernel(opts_vs, hw_info, sizes, problem_vs, reqs_vs, adjust_vs);
    } catch (const std::runtime_error& ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create VS sdpa_micro kernel: " << ex.what() << "\n";
        throw;
    }
}
}  // namespace ov::intel_gpu::ocl
#endif
