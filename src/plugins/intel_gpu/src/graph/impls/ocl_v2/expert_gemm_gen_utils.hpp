// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/// Utilities shared by GatherMatMul and GroupedMatMul batched-GEMM generators.

#ifdef ENABLE_ONEDNN_FOR_GPU

#    include <oneapi/dnnl/dnnl.hpp>

#    include "gather_matmul/gather_matmul_gen_micro.hpp"
#    include "intel_gpu/graph/kernel_impl_params.hpp"
#    include "intel_gpu/primitives/swiglu.hpp"
#    include "intel_gpu/runtime/device_info.hpp"
#    include "ocl_v2/utils/jitter.hpp"
#    include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Subgroup size for the expert GEMM micro-kernel (same for both operations).
inline size_t get_expert_subgroup_size(gpu_arch arch) {
    switch (arch) {
    case gpu_arch::gen9:
    case gpu_arch::gen11:
    case gpu_arch::xe_lp:
    case gpu_arch::xe_hp:
    case gpu_arch::xe_hpg:
        return 8;
    case gpu_arch::xe2:
    case gpu_arch::xe3:
        return 16;
    default:
        return 0;
    }
}

/// Returns true when a fused SwiGLU primitive is attached to the node.
inline bool has_fused_swiglu(const kernel_impl_params& params) {
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<swiglu>())
            return true;
    }
    return false;
}

/// cldnn → oneDNN element-type conversion (oneDNN execution path).
inline dnnl::memory::data_type to_onednn_dtype(cldnn::data_types dt) {
    switch (dt) {
    case cldnn::data_types::f32: return dnnl::memory::data_type::f32;
    case cldnn::data_types::f16: return dnnl::memory::data_type::f16;
    case cldnn::data_types::i8:  return dnnl::memory::data_type::s8;
    case cldnn::data_types::u8:  return dnnl::memory::data_type::u8;
    case cldnn::data_types::i32: return dnnl::memory::data_type::s32;
    case cldnn::data_types::i4:  return dnnl::memory::data_type::s4;
    case cldnn::data_types::u4:  return dnnl::memory::data_type::u4;
    default:
        throw std::invalid_argument("[GPU] expert GEMM: unsupported cldnn->onednn type conversion");
    }
}

/// Add weight-quantisation JIT constants (EXPERT_STRIDE, WEIGHT_COMPRESSED_INT4, scales, ZPs).
/// `weight_input_idx` is the primitive input index of the weight tensor.
inline void add_expert_weight_quant_jit(JitConstants& jit,
                                         const kernel_impl_params& params,
                                         const gathermatmul_config& cfg,
                                         size_t weight_input_idx) {
    const auto& weight_shape = params.input_layouts[weight_input_idx].get_shape();
    const bool is_u4_i4 = (params.input_layouts[weight_input_idx].data_type == cldnn::data_types::u4 ||
                            params.input_layouts[weight_input_idx].data_type == cldnn::data_types::i4);

    if (cfg.is_weight_quantized) {
        const auto& scale_shape = params.input_layouts[cfg.weight_scale_idx].get_shape();

        jit.make("WEIGHT_SCALE_DT", to_ocl_type(cldnn::data_types::f16));
        jit.make("NUM_GROUPS", (cfg.weight_group_size > 0) ? scale_shape[2] : size_t{1});
        jit.make("SCALE_ZP_NO_TRANSPOSE", 1);

        const size_t expert_stride = weight_shape.size() == 4
                                         ? weight_shape[1] * weight_shape[2] * weight_shape[3]
                                         : weight_shape[1] * weight_shape[2];
        if (is_u4_i4) {
            jit.make("EXPERT_STRIDE", expert_stride / 2);
            jit.make("WEIGHT_COMPRESSED_INT4", 1);
        } else {
            jit.make("EXPERT_STRIDE", expert_stride);
        }

        if (!cfg.is_weight_symmetric_quantized) {
            const auto& zp_layout = params.input_layouts[cfg.weight_zp_idx];
            const bool zp_u4_i4 = (zp_layout.data_type == cldnn::data_types::u4 ||
                                    zp_layout.data_type == cldnn::data_types::i4);
            if (zp_u4_i4) {
                jit.make("WEIGHT_COMPRESSED_ZP_INT4", 1);
                jit.make("WEIGHT_ZP_DT", to_ocl_type(cldnn::data_types::u8));
            } else {
                jit.make("WEIGHT_ZP_DT", to_ocl_type(cldnn::data_types::f16));
            }
        }
    } else {
        jit.make("EXPERT_STRIDE", weight_shape[1] * weight_shape[2]);
    }
}

/// Add SwiGLU-fusion JIT constants (SWIGLU_FUSED, SWIGLU_LENGTH, SWIGLU_GATE_IDX, SWISH_BETA).
/// `weight_n_dim` is the full N dimension of the weight tensor (halved by fused SwiGLU).
inline void add_swiglu_jit(JitConstants& jit,
                            const kernel_impl_params& params,
                            size_t weight_n_dim) {
    if (!has_fused_swiglu(params))
        return;
    jit.make("SWIGLU_FUSED", 1);
    jit.make("SWIGLU_LENGTH", weight_n_dim / 2);
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<swiglu>()) {
            auto desc = fd.typed_desc<swiglu>();
            jit.make("SWIGLU_GATE_IDX", desc->gate_idx);
            jit.make("SWISH_BETA", desc->swish_beta);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Base class for expert GEMM batched generators
// ---------------------------------------------------------------------------

/// Common base for GatherMatmulBatchedGemmGenerator and GroupedMatmulBatchedGemmGenerator.
///
/// Implements:
///   - get_build_options   — adds the 4 required Intel extensions
///   - get_kernel_data     — template-method that calls build_jit_constants (pure virtual)
///
/// Subclasses implement:
///   - build_jit_constants — kernel-specific layout JIT + dispatch hints
///   - get_arguments_desc  — kernel-specific argument list
///   - get_dispatch_data_func — kernel-specific work-group sizing
class ExpertGemmBatchedGeneratorBase : public KernelGenerator {
public:
    using KernelGenerator::KernelGenerator;

    // ---- shared: get_build_options ----------------------------------------

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override {
        auto base = KernelGenerator::get_build_options(params);
        base += " -Dcl_intel_dot_accumulate";
        base += " -Dcl_intel_global_float_atomic";
        base += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
        base += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
        return base;
    }

    // ---- shared: get_jit_constants (1-arg) is unused; 3-arg version in subclasses ----

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& /*params*/) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }

    // ---- shared: get_kernel_data (template method) -------------------------

    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override {
        micro::Package bgm_gemm;
        const auto& device_info = params.get_device_info();
        try {
            GatherMatmulMicroGenerator::init_microkernels(params, bgm_gemm, /*is_prefill=*/true);
        } catch (const std::runtime_error& ex) {
            OPENVINO_THROW(get_class_name(), "::get_kernel_data() - can't init microkernels: ", ex.what());
        }

        auto cfg = GatherMatmulMicroGenerator::get_config(params);
        auto jit = build_jit_constants(params, bgm_gemm, cfg);

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

        // Generate gemmstone micro-kernel shim (always "gm" decorator).
        micro::ShimOptions shim_options;
        shim_options.subgroupSize = static_cast<int32_t>(get_expert_subgroup_size(device_info.arch));
        shim_options.useTileOps = true;
        shim_options.decorator = "gm";
        kd.code->jit += generateShim(bgm_gemm, micro::HostLanguage::OpenCL_C, shim_options);
        if (bgm_gemm.grfMin > 128)
            kd.code->options += " -cl-intel-256-GRF-per-thread";

        kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(bgm_gemm));

        const uint32_t slm_size = kd.micro_kernels[0]->p.getSetting("slm_size");
        kd.params.local_memory_args.clear();
        if (slm_size > 0) {
            kd.params.local_memory_args.push_back(slm_size);
            kd.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
        }
        return kd;
    }

protected:
    /// Pure virtual: subclass builds the kernel-specific JIT constants.
    [[nodiscard]] virtual JitConstants build_jit_constants(const kernel_impl_params& params,
                                                            const micro::Package& bgm_gemm,
                                                            const gathermatmul_config& cfg) const = 0;

    /// Name used in error messages — override in each subclass.
    [[nodiscard]] virtual const char* get_class_name() const = 0;
};

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_ONEDNN_FOR_GPU
