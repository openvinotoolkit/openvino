// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "micro_utils.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "utils/kernel_generator.hpp"

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

inline int64_t get_d_max(int64_t head_size) {
    for (int64_t i = 32; i <= 1024; i *= 2) {
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
    default:
        break;
    }
    OPENVINO_THROW("Unsupported element type: ", t);
}

ov::Dimension get_seq_length(const layout& qkv, const std::vector<int64_t>& order) {
    return qkv.get_partial_shape()[order[2]];
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

sdpa_config_t xehpg_q_h64 = {32, 16, 16, 16, 4, 4, 4, 4};
sdpa_config_t xehpg_q_h64_2nd = {16, 16, 8, 8, 16, 1, 8, 2};

sdpa_config_t xehpg_h128 = {16, 16, 32, 8, 8, 4, 4, 8};
sdpa_config_t xehpg_h128_s32 = {16, 16, 16, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h128_2nd = {8, 16, 16, 8, 16, 1, 8, 2};
sdpa_config_t xehpg_h128_s256_2nd = {8, 16, 32, 8, 8, 1, 4, 2};

sdpa_config_t xehpg_q_h128 = {32, 16, 16, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_q_h128_2nd = {32, 16, 16, 8, 16, 1, 8, 2};
sdpa_config_t xehpg_q_h128_s64_2nd = {16, 16, 16, 8, 16, 1, 8, 2};

sdpa_config_t xehpg_h256 = {16, 16, 32, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h256_s128 = {8, 16, 32, 16, 8, 4, 8, 4};
sdpa_config_t xehpg_h256_s32 = {8, 16, 32, 8, 16, 2, 8, 4};
sdpa_config_t xehpg_h256_2nd = {8, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s64_2nd = {16, 8, 16, 8, 16, 1, 16, 1};
sdpa_config_t xehpg_h256_s32_2nd = {16, 16, 32, 8, 16, 1, 8, 2};

sdpa_config_t xehpc_h32 = {16, 64, 32, 16, 4, 2, 1, 8};
sdpa_config_t xehpc_h32_s32 = {16, 16, 16, 16, 2, 4, 2, 4};
sdpa_config_t xehpc_h32_2nd = {16, 64, 16, 16, 8, 1, 2, 4};

sdpa_config_t xehpc_h64 = {16, 64, 32, 16, 8, 2, 2, 8};
sdpa_config_t xehpc_h64_s64 = {32, 32, 32, 16, 4, 2, 2, 4};
sdpa_config_t xehpc_h64_s32 = {16, 16, 16, 16, 4, 2, 4, 2};
sdpa_config_t xehpc_h64_2nd = {32, 32, 32, 16, 4, 1, 2, 2};
sdpa_config_t xehpc_h64_s64_2nd = {16, 16, 16, 16, 4, 1, 4, 1};

sdpa_config_t xehpc_q_h64 = {16, 64, 32, 16, 8, 4, 2, 16};

sdpa_config_t xehpc_h128 = {16, 64, 32, 16, 16, 2, 4, 8};
sdpa_config_t xehpc_h128_s64 = {16, 32, 32, 32, 4, 2, 4, 2};
sdpa_config_t xehpc_h128_s32 = {16, 16, 16, 16, 8, 2, 8, 2};
sdpa_config_t xehpc_h128_2nd = {32, 32, 32, 16, 8, 1, 4, 2};

sdpa_config_t xehpc_q_h128 = {16, 64, 16, 32, 16, 2, 8, 4};
sdpa_config_t xehpc_q_h128_s64 = {16, 16, 32, 16, 4, 4, 4, 4};
sdpa_config_t xehpc_q_h128_s32 = {16, 16, 32, 16, 4, 2, 4, 2};
sdpa_config_t xehpc_q_h128_2nd = {32, 32, 16, 32, 4, 1, 4, 1};
sdpa_config_t xehpc_q_h128_s32_2nd = {16, 32, 16, 16, 8, 1, 4, 2};

sdpa_config_t xehpc_h256 = {16, 32, 32, 32, 8, 4, 8, 4};
sdpa_config_t xehpc_h256_s64 = {16, 32, 32, 32, 8, 1, 8, 1};
sdpa_config_t xehpc_h256_2nd = {16, 16, 16, 16, 16, 1, 16, 1};

sdpa_config_t* choose_config_xehpg(int head_size, int seq, bool thin_q, bool quantized) {
    if (head_size <= 32) {
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
        if (quantized) {
            if (thin_q)
                return &xehpg_q_h64_2nd;
            return &xehpg_q_h64;
        }
        if (thin_q)
            return &xehpg_h64_2nd;
        if (seq <= 64)
            return &xehpg_h64_s64;
        if (seq <= 128)
            return &xehpg_h64_s128;
        return &xehpg_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 64)
                    return &xehpg_q_h128_s64_2nd;
                return &xehpg_q_h128_2nd;
            }
            if (seq <= 32)
                return &xehpg_h128_s32;
            return &xehpg_q_h128;
        }
        if (thin_q) {
            if (seq <= 256)
                return &xehpg_h128_s256_2nd;
            return &xehpg_h128_2nd;
        }
        if (seq <= 32)
            return &xehpg_h128_s32;
        return &xehpg_h128;
    } else if (head_size <= 256) {
        if (thin_q) {
            if (seq <= 32)
                return &xehpg_h256_s32_2nd;
            if (seq <= 64)
                return &xehpg_h256_s64_2nd;
            return &xehpg_h256_2nd;
        }
        if (seq <= 32)
            return &xehpg_h256_s32;
        if (seq <= 128)
            return &xehpg_h256_s128;
        return &xehpg_h256;
    }
    return nullptr;
}

sdpa_config_t* choose_config_xehpc(int head_size, int seq, bool thin_q, bool quantized) {
    if (head_size <= 32) {
        if (thin_q)
            return &xehpc_h32_2nd;
        if (seq <= 32)
            return &xehpc_h32_s32;
        return &xehpc_h32;
    } else if (head_size <= 64) {
        if (thin_q) {
            if (seq <= 64)
                return &xehpc_h64_s64_2nd;
            return &xehpc_h64_2nd;
        }
        if (quantized && seq >= 256)
            return &xehpc_q_h64;
        if (seq <= 32)
            return &xehpc_h64_s32;
        if (seq <= 64)
            return &xehpc_h64_s64;
        return &xehpc_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 32)
                    return &xehpc_q_h128_s32_2nd;
                return &xehpc_q_h128_2nd;
            }
            if (seq <= 32)
                return &xehpc_q_h128_s32;
            if (seq <= 64)
                return &xehpc_q_h128_s64;
            return &xehpc_q_h128;
        }
        if (thin_q)
            return &xehpc_h128_2nd;
        if (seq <= 32)
            return &xehpc_h128_s32;
        if (seq <= 64)
            return &xehpc_h128_s64;
        return &xehpc_h128;
    } else if (head_size <= 256) {
        if (thin_q)
            return &xehpc_h256_2nd;
        if (seq <= 64)
            return &xehpc_h256_s64;
        return &xehpc_h256;
    }
    return nullptr;
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

KernelData SDPAMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    std::vector<micro::Package> gemms(2);  // KQ and VS
    init_microkernels(params, gemms[kq_id], gemms[vs_id], m_is_prefill);

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

JitConstants SDPAMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const {
    auto jit = SDPABase::get_jit_constants(params);
    jit.add(make_tensors_jit_constants(params));
    auto desc = params.typed_desc<scaled_dot_product_attention>();
    const auto& device_info = params.get_device_info();

    const auto& Q = params.input_layouts[0];
    const auto& K = params.input_layouts[1];
    const auto& V = params.input_layouts[2];
    const auto& out = params.output_layouts[0];
    const auto& out_ps = out.get_partial_shape();

    const auto head_size = Q.get_partial_shape()[3].get_length();
    const auto d_max = get_d_max(head_size);
    const ov::Dimension n_keys = get_seq_length(K, desc->input_k_transpose_order);
    const ov::Dimension n_queries = get_seq_length(Q, desc->input_q_transpose_order);
    const ov::Dimension n_values = V.get_partial_shape()[3];
    const auto batch = out_ps[0] * out_ps[1];

    auto ldq = head_size * ov::element::Type(Q.data_type).size();
    auto ldk = head_size * ov::element::Type(K.data_type).size();
    auto ldv = head_size * ov::element::Type(V.data_type).size();
    auto lda = head_size * ov::element::Type(out.data_type).size();

    jit.make("D_MAX", d_max);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));
    jit.make("INVERT_SCALE", false);
    jit.make("SCALE_DATA_T", "half");

    auto data_inputs_num = get_data_inputs_num(*desc);

    jit.make("WITH_ATTN_MASK", data_inputs_num > 3);
    if (data_inputs_num > 3) {
        jit.add(convert_strides("MSK", "INPUT3", {0, 1, 2, 3}));
        jit.add(unit_parameters("MSK"));
    }
    jit.make("WITH_SCALE", data_inputs_num > 4);
    jit.make("Q_ALIGN", micro::alignment_for_ld(ldq));
    jit.make("K_ALIGN", micro::alignment_for_ld(ldk));
    jit.make("V_ALIGN", micro::alignment_for_ld(ldv));
    jit.make("A_ALIGN", micro::alignment_for_ld(lda));

    jit.make("TRANSPOSE_K", false);

    jit.make("QRY_DATA_T", to_ocl_type(Q.data_type));
    jit.make("KEY_DATA_T", to_ocl_type(K.data_type));
    jit.make("VAL_DATA_T", to_ocl_type(V.data_type));

    const bool use_asymmetric_quantization = desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;

    auto elems_per_byte = [](ov::element::Type dt) {
        switch (dt) {
        case ov::element::u4:
        case ov::element::i4:
            return 2;
        default:
            return 1;
        }
    };

    if (desc->is_kv_compressed) {
        const auto& key_cache_comp_scale = params.input_layouts[data_inputs_num];
        const auto& value_cache_comp_scale = params.input_layouts[data_inputs_num + 1];
        jit.make("KV_COMPRESSED", 1);
        jit.make("KEY_ATTR_SCALES_DATA_T", to_ocl_type(key_cache_comp_scale.data_type));
        jit.make("VAL_ATTR_SCALES_DATA_T", to_ocl_type(value_cache_comp_scale.data_type));

        int kq_scale_mask = (static_cast<int>(desc->is_kv_compressed) << 1) | static_cast<int>(kq_common_scales);
        int vs_scale_mask = (static_cast<int>(desc->is_kv_compressed) << 1) | static_cast<int>(vs_common_scales);
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

    bool d_full = (head_size == d_max);
    bool v_full = (head_size == tile_v);
    bool k_full = !n_keys.is_dynamic() && (n_keys.get_length() % tile_k) == 0;
    bool q_full = !n_queries.is_dynamic() && (n_queries.get_length() % tile_q) == 0;

    auto Q_num_heads_dim = get_num_heads(Q, desc->input_q_transpose_order);
    auto K_num_heads_dim = get_num_heads(K, desc->input_k_transpose_order);

    jit.make("REMAINDER_K", !k_full);
    jit.make("KV_GROUP_SIZE", Q_num_heads_dim.get_length() / K_num_heads_dim.get_length());

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

    jit.add(convert_strides("QRY", "INPUT0", desc->input_q_transpose_order));
    jit.add(convert_strides("KEY", "INPUT1", desc->input_k_transpose_order));
    jit.add(convert_strides("VAL", "INPUT2", desc->input_v_transpose_order));
    jit.add(convert_strides("DST", "OUTPUT", desc->output_transpose_order));

    jit.add(unit_parameters("QRY"));
    jit.add(unit_parameters("KEY"));
    jit.add(unit_parameters("VAL"));
    jit.add(unit_parameters("DST"));

    return jit;
}

Arguments SDPAMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    auto desc = params.typed_desc<scaled_dot_product_attention>();

    auto data_inputs_num = get_data_inputs_num(*desc);

    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // K
    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // Q
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // V

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // A

    if (data_inputs_num >= 4)
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // mask
    if (data_inputs_num >= 5)
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // Scale

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // D
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // K
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // Q

    if (desc->is_kv_compressed) {
        const bool is_asym_quantization = desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
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
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(3);
        if (!params.is_dynamic()) {
            auto desc = params.typed_desc<scaled_dot_product_attention>();
            const auto& device_info = params.get_device_info();
            const auto& gemms = kd.micro_kernels;
            const auto& gemm_kq = gemms[kq_id]->p;

            const auto& Q = params.input_layouts[0];
            const auto& K = params.input_layouts[1];
            const auto& V = params.input_layouts[2];
            const auto& out = params.output_layouts[0];
            const auto& out_ps = out.get_partial_shape();

            const auto head_size = Q.get_partial_shape()[3].get_length();
            const ov::Dimension n_keys = get_seq_length(K, desc->input_k_transpose_order);
            const ov::Dimension n_queries = get_seq_length(Q, desc->input_q_transpose_order);
            const ov::Dimension n_values = V.get_partial_shape()[3];

            auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
            auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m") * gemm_kq.getSetting("sg_per_wg_n");

            wgs.local = {get_subgroup_size(device_info.arch), (size_t)sg_per_wg, 1};
            wgs.global = wgs.local;

            wgs.global[0] *= ceil_div(n_queries.get_length(), wg_tile_q);
            wgs.global[1] *= out_ps[1].get_length();
            wgs.global[2] *= out_ps[0].get_length();

            ScalarDescriptor s_d{ScalarDescriptor::Types::INT32};
            s_d.v.s32 = static_cast<uint32_t>(head_size);
            scalars.push_back(s_d);

            ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
            s_k.v.s32 = static_cast<uint32_t>(n_keys.get_length());
            scalars.push_back(s_k);

            ScalarDescriptor s_q{ScalarDescriptor::Types::INT32};
            s_q.v.s32 = static_cast<uint32_t>(n_queries.get_length());
            scalars.push_back(s_q);
        }
    }};
}

void SDPAMicroGenerator::init_microkernels(const kernel_impl_params& params, micro::Package& gemm_kq, micro::Package& gemm_vs, bool is_prefill) {
    // TODO: Remove once micro API is thread safe
    static std::mutex m;
    std::lock_guard l(m);

    const auto& Q = params.input_layouts[0];
    const auto& K = params.input_layouts[1];
    const auto& V = params.input_layouts[2];
    const auto& out = params.output_layouts[0];
    const auto& out_ps = out.get_partial_shape();

    auto desc = params.typed_desc<scaled_dot_product_attention>();
    const auto& device_info = params.get_device_info();

    const auto head_size = Q.get_partial_shape()[3].get_length();
    const auto d_max = get_d_max(head_size);
    const ov::Dimension n_keys = get_seq_length(K, desc->input_k_transpose_order);
    const ov::Dimension n_queries = get_seq_length(Q, desc->input_q_transpose_order);
    const ov::Dimension n_values = V.get_partial_shape()[3];
    const auto batch = out_ps[0] * out_ps[1];

    /* Retrieve pre-tuned kernel configuration */
    sdpa_config_t* config = nullptr;
    bool thin_q = (!n_queries.is_dynamic() && (n_queries.get_length() <= 16)) || !is_prefill;

    bool is_quantized =
        (K.data_type == ov::element::u8 || K.data_type == ov::element::i8) || (V.data_type == ov::element::u8 || V.data_type == ov::element::i8);

    int32_t nkeys_v = n_keys.is_dynamic() ? 0 : n_keys.get_length();
    switch (device_info.arch) {
    case gpu_arch::xe_hpg: {
        config = choose_config_xehpg(static_cast<int32_t>(head_size), nkeys_v, thin_q, is_quantized);
        break;
    }
    case gpu_arch::xe_hpc:
    case gpu_arch::xe2:
    case gpu_arch::xe3: {
        config = choose_config_xehpc(static_cast<int32_t>(head_size), nkeys_v, thin_q, is_quantized);
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

    const bool use_asymmetric_quantization = desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;

    const auto data_inputs_num = get_data_inputs_num(*desc);

    if (desc->is_kv_compressed && !kq_common_scales) {
        const auto& key_cache_comp_scale = params.input_layouts[data_inputs_num];

        const auto scale_dt = convert_type(key_cache_comp_scale.data_type);
        problem_kq.Ta_scale = scale_dt;
        problem_kq.A_scale.alignment = micro::data_type_size(scale_dt);

        problem_kq.A_scale.layout = micro::MatrixLayout::T;
        problem_kq.aScale2D = true;
    }

    if (desc->is_kv_compressed && use_asymmetric_quantization) {
        const auto& key_cache_comp_zp = params.input_layouts[data_inputs_num + 2];
        const auto zp_dt = convert_type(key_cache_comp_zp.data_type);
        problem_kq.Tao = zp_dt;
        problem_kq.AO.alignment = micro::data_type_size(zp_dt);
        problem_kq.AO.layout = micro::MatrixLayout::T;
        problem_kq.aoPtrDims = kq_common_zp ? 0 : 2;
        problem_kq.aOffset = micro::ABOffset::Calc;
    }

    if (desc->is_kv_compressed) {
        problem_kq.aqGroupM = 1;
        problem_kq.aqGroupK = (kq_common_scales || kq_common_zp) ? 1 : head_size;
    }

    opts_kq.scaleA = desc->is_kv_compressed && !kq_common_scales;
    opts_kq.offsetA = desc->is_kv_compressed && use_asymmetric_quantization;

    problem_kq.B.layout = micro::MatrixLayout::Pr;
    problem_kq.C.layout = micro::MatrixLayout::T;
    problem_kq.A.setAlignment(micro::alignment_for_ld(head_size * problem.Ta));
    problem_kq.B.setAlignment(64);  // Q is packed in VNNI format in SLM
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = d_max;
    problem_kq.B.tileC = static_cast<uint16_t>(get_subgroup_size(device_info.arch));

    /* Set up problem size information */
    micro::SizeParams sizes;
    sizes.m = n_keys.is_dynamic() ? 0 : n_keys.get_length();
    sizes.n = n_queries.is_dynamic() ? 0 : n_queries.get_length();
    sizes.k = head_size;
    sizes.batch = batch.is_dynamic() ? 0 : batch.get_length();

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

    if (desc->is_kv_compressed && !vs_common_scales) {
        const auto& value_cache_comp_scale = params.input_layouts[data_inputs_num + 1];
        auto scale_dt = convert_type(value_cache_comp_scale.data_type);
        problem_vs.Ta_scale = scale_dt;
        problem_vs.A_scale.alignment = micro::data_type_size(scale_dt);
        problem_vs.A_scale.layout = micro::MatrixLayout::N;
        problem_vs.aScale2D = true;
    }

    if (desc->is_kv_compressed && use_asymmetric_quantization) {
        const auto& value_cache_comp_zp = params.input_layouts[data_inputs_num + 3];
        auto zp_dt = convert_type(value_cache_comp_zp.data_type);
        problem_vs.Tao = zp_dt;
        problem_vs.AO.alignment = micro::data_type_size(zp_dt);
        problem_vs.AO.layout = micro::MatrixLayout::N;
        problem_vs.aoPtrDims = vs_common_zp ? 0 : 2;
        problem_vs.aOffset = micro::ABOffset::Calc;
    }

    if (desc->is_kv_compressed) {
        problem_vs.aqGroupM = (vs_common_scales || vs_common_zp) ? 1 : micro::rnd_up_pow2(head_size);
        problem_vs.aqGroupK = 1;
    }

    opts_vs.scaleA = desc->is_kv_compressed && !vs_common_scales;
    opts_vs.offsetA = desc->is_kv_compressed && use_asymmetric_quantization;

    problem_vs.B.layout = micro::MatrixLayout::Pr;
    problem_vs.C.layout = micro::MatrixLayout::N;
    problem_vs.A.setAlignment(micro::alignment_for_ld(head_size * problem.Ta));
    problem_vs.B.setAlignment(64);  // S is packed in SLM
    problem_vs.B.crosspack = 16;
    sizes.m = n_values.is_dynamic() ? 0 : n_values.get_length();
    sizes.n = gemm_kq.getSetting("wg_tile_n");
    sizes.k = gemm_kq.getSetting("wg_tile_m");

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
