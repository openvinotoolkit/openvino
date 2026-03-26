// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
#include "gather_matmul_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "gather_matmul_inst.h"
#include "../utils/kernel_generator.hpp"
#include "gemmstone/kernel_selector.hpp"

// clang-format on
namespace ov::intel_gpu::ocl {

namespace {
void entryObserver(const gemmstone::kcatalog::Entry* entry, double score, gemmstone::EvaluateAuxOutput aux) {
    GPU_DEBUG_TRACE_DETAIL << "GatherMatmulconsider strategy: " << entry->str() << ", score: " << score << "\n";
};
}  // anonymous namespace

static bool has_fused_swiglu(const kernel_impl_params& params) {
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<swiglu>())
            return true;
    }
    return false;
}

static size_t get_subgroup_size(gpu_arch arch) {
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

gathermatmul_config GatherMatmulMicroGenerator::get_config(const kernel_impl_params& params) {
    gathermatmul_config cfg;
    auto desc = params.typed_desc<gather_matmul>();
    std::vector<cldnn::data_types> quantized_types = {data_types::u4, data_types::i4, data_types::u8, data_types::i8};
    cfg.has_bias = desc->has_bias;

    if (std::any_of(quantized_types.begin(), quantized_types.end(), [&](const cldnn::data_types& t) -> bool {
            return t == params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].data_type;
        })) {
        cfg.is_weight_quantized = true;
        // GatherMatmulCompressed always has all 6 inputs (A, B, indices, bias_placeholder, scales, zp),
        // even when has_bias=false. So scale/zp indices are always at their fixed positions.
        cfg.weight_scale_idx = gather_matmul::BGMInputIdx::WEIGHT_SCALE;
        cfg.weight_zp_idx = gather_matmul::BGMInputIdx::WEIGHT_ZP;
        const auto& weight_shape = params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].get_shape();
        auto k = (weight_shape.size() == 4) ? weight_shape[2] * weight_shape[3] : weight_shape[2];
        // Scale shape is [E, N, G] or [E, N, G, 1] — num groups is always at index 2
        const auto& scale_shape_for_groups = params.input_layouts[cfg.weight_scale_idx].get_shape();
        auto num_scale_groups = scale_shape_for_groups[2];
        cfg.weight_group_size = k / num_scale_groups;
        // GatherMatmulCompressed always has 6 inputs (including ZP placeholder).
        // Check if ZP is a real input (non-empty) to determine symmetric quantization.
        if (static_cast<int32_t>(params.input_layouts.size()) > cfg.weight_zp_idx && params.input_layouts[cfg.weight_zp_idx].count() > 0) {
            cfg.is_weight_symmetric_quantized = false;
        } else {
            cfg.is_weight_symmetric_quantized = true;
        }
    }
    return cfg;
}

JitConstants GatherMatmulMicroGenerator::get_jit_constants(const kernel_impl_params& params,
                                                           const micro::Package& bgm_gemm,
                                                           const gathermatmul_config& cfg) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));

    // GatherMatmulinput order: INPUT=0, WEIGHT=1, INDICES=2, BIAS=3, SCALE=4, ZP=5
    std::vector<size_t> input_ids = {
        gather_matmul::BGMInputIdx::INPUT,
        gather_matmul::BGMInputIdx::WEIGHT,
        gather_matmul::BGMInputIdx::INDICES,
    };

    bool is_u4_i4 = (params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].data_type == data_types::u4 ||
                     params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].data_type == data_types::i4);
    auto weight_idx = gather_matmul::BGMInputIdx::WEIGHT;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();

    if (cfg.is_weight_quantized) {
        auto scale_idx = cfg.weight_scale_idx;
        const auto& scale_shape = params.input_layouts[scale_idx].get_shape();

        if (cfg.has_bias) {
            auto bias_idx = gather_matmul::BGMInputIdx::BIAS;
            const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
            input_ids.push_back(gather_matmul::BGMInputIdx::BIAS);
        }

        input_ids.push_back(static_cast<size_t>(scale_idx));
        if (!cfg.is_weight_symmetric_quantized)
            input_ids.push_back(static_cast<size_t>(cfg.weight_zp_idx));

        jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
        jit.make("SCALE_ZP_NO_TRANSPOSE", 1);
        if (cfg.weight_group_size > 0)
            jit.make("NUM_GROUPS", scale_shape[2]);
        else
            jit.make("NUM_GROUPS", 1);

        size_t expert_stride = weight_shape.size() == 4 ? (weight_shape[1] * weight_shape[2] * weight_shape[3]) : (weight_shape[1] * weight_shape[2]);
        if (is_u4_i4) {
            jit.make("EXPERT_STRIDE", expert_stride / 2);
            jit.make("WEIGHT_COMPRESSED_INT4", 1);
        } else {
            jit.make("EXPERT_STRIDE", expert_stride);
        }
        if (!cfg.is_weight_symmetric_quantized) {
            const auto& zp_layout = params.input_layouts[cfg.weight_zp_idx];
            bool is_zp_u4_i4 = (zp_layout.data_type == data_types::u4 || zp_layout.data_type == data_types::i4);
            if (is_zp_u4_i4) {
                jit.make("WEIGHT_COMPRESSED_ZP_INT4", 1);
                jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::u8));
            } else {
                jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::f16));
            }
        }
    } else {
        jit.make("EXPERT_STRIDE", weight_shape[1] * weight_shape[2]);
        if (cfg.has_bias) {
            auto bias_idx = gather_matmul::BGMInputIdx::BIAS;
            const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
            input_ids.push_back(gather_matmul::BGMInputIdx::BIAS);
        }
    }

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

    // INPUT_STRIDE = K dimension of weights (stride between rows in weight matrix)
    jit.make("INPUT_STRIDE", weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2]);
    // OUTPUT_STRIDE = output features (N after SwiGLU, or weight_shape[1] otherwise)
    bool fused_swiglu = has_fused_swiglu(params);
    size_t m_out = fused_swiglu ? weight_shape[1] / 2 : weight_shape[1];
    jit.make("OUTPUT_STRIDE", m_out);

    if (fused_swiglu) {
        jit.make("SWIGLU_FUSED", 1);
        jit.make("SWIGLU_LENGTH", m_out);
        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<swiglu>()) {
                auto swiglu_desc = fd.typed_desc<swiglu>();
                jit.make("SWIGLU_GATE_IDX", swiglu_desc->gate_idx);
                jit.make("SWISH_BETA", swiglu_desc->swish_beta);
                break;
            }
        }
    }

    // GatherMatmul-specific constants — use LayoutJitter for dynamic-shape-safe access
    // INPUT0 = activations [n_activated_experts, n_tokens, hidden_size] (rank 3: BATCH=dim0, FEATURE=dim1, Y=dim2)
    // INPUT2 = indices [n_tokens, top_k] (rank 2: BATCH=dim0, FEATURE=dim1)
    // For dynamic dims, LayoutJitter generates shape_info[offset] references; for static, literals.
    jit.make("N_TOKENS", "INPUT0_FEATURE_NUM");
    jit.make("N_ACTIVATED_EXPERTS", "INPUT0_BATCH_NUM");
    jit.make("TOP_K", "INPUT2_FEATURE_NUM");

    if (!m_is_prefill)
        jit.make("IS_GENERATE", 1);

    auto slm_size = bgm_gemm.getSetting("slm_size");
    if (slm_size > 0)
        jit.make("USE_SLM", 1);
    return jit;
}

static micro::Type convert_type(ov::element::Type t) {
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
    case ov::element::u4:
        return micro::Type::u4;
    case ov::element::i4:
        return micro::Type::s4;
    default:
        break;
    }
    OPENVINO_THROW("Unsupported element type: ", t);
}

std::mutex GatherMatmulMicroGenerator::mtx;

void GatherMatmulMicroGenerator::init_microkernels(const kernel_impl_params& params, micro::Package& gemm_bgm, bool is_prefill) {
    std::lock_guard<std::mutex> l(mtx);
    auto bgm_cfg = get_config(params);
    const auto& device_info = params.get_device_info();
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;

    // B (weights): [n_all_experts, N, K] — N=ofm, K=ifm
    const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
    size_t m = weight_shape[1];      // N (output features)
    size_t n = is_prefill ? 32 : 8;  // token count hint
    size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
    GPU_DEBUG_TRACE_DETAIL << "GatherMatmulinit_microkernels for " << (is_prefill ? "prefill" : "generate") << " : Seq_len:" << n << " Ofm:" << m << " K:" << k
                           << "\n";

    micro::GEMMProblem problem;
    micro::GEMMOptions opts;
    opts.slmPtr = true;
    opts.kParallelLocal = !is_prefill;
    enum class MICRO_DIMENSIONALITY { NONE = -1, SCALAR = 0, VECTOR = 1, MATRIX = 2 };

    if (bgm_cfg.is_weight_quantized) {
        problem.Ta = micro::Type::f16;
        problem.Ta_ext = convert_type(params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).data_type);
        problem.A.setAlignment(micro::alignment_for_ld(k * problem.Ta_ext));

        problem.Ta_scale = convert_type(params.get_input_layout(bgm_cfg.weight_scale_idx).data_type);
        problem.A_scale.setAlignment(2);
        problem.A_scale.layout = micro::MatrixLayout::N;
        problem.asPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);

        problem.aqGroupM = 1;
        problem.aqGroupK = (bgm_cfg.weight_group_size == -1) ? static_cast<int32_t>(k) : bgm_cfg.weight_group_size;

        opts.scaleA = true;
        if (!bgm_cfg.is_weight_symmetric_quantized) {
            const auto& zp_layout = params.get_input_layout(bgm_cfg.weight_zp_idx);
            const auto zp_dt = convert_type(zp_layout.data_type);
            problem.Tao = zp_dt;
            problem.AO.setAlignment(zp_dt == gemmstone::Type::u4 ? 1 : static_cast<int32_t>(zp_dt.size()));
            problem.AO.layout = micro::MatrixLayout::N;
            problem.aoPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);
            problem.aOffset = micro::ABOffset::Calc;
            opts.offsetA = true;
        }
    } else {
        problem.Ta = problem.Ta_ext = convert_type(params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).data_type);
        problem.A.setAlignment(micro::alignment_for_ld(k * problem.Ta));
    }

    problem.Tb = problem.Tb_ext = micro::Type::f16;
    problem.Tc = micro::Type::f32;
    problem.Tc_ext = micro::Type::f32;
    problem.Ts = problem.Tc;
    problem.A.layout = micro::MatrixLayout::T;
    problem.B.layout = micro::MatrixLayout::N;
    problem.C.layout = micro::MatrixLayout::N;
    problem.B.setAlignment(micro::alignment_for_ld(k * problem.Tb));
    problem.C.setAlignment(static_cast<int32_t>(problem.Tc.size()));

    micro::SizeParams sizes;
    sizes.n = static_cast<int32_t>(n);
    sizes.m = static_cast<int32_t>(m);
    sizes.k = static_cast<int32_t>(k);
    sizes.batch = 1;

    GPU_DEBUG_TRACE_DETAIL << "GatherMatmulproblem:" << problem.toString() << "\n";
    GPU_DEBUG_TRACE_DETAIL << "GatherMatmulsizes: m=" << m << " n=" << n << " k=" << k << std::endl;
    try {
        gemmstone::SelectionObserver observer = entryObserver;
        gemm_bgm = micro::select_gemm_microkernel(opts, hw_info, sizes, problem, &observer);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("Can't create GatherMatmul micro kernel: ", ex.what());
    }
}

DispatchDataFunc GatherMatmulMicroGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_per_wg_k = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_k"));
        auto wg_tile_m = gemm_p.getSetting("wg_tile_m");

        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(2);

        auto weight_layout = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT);
        const auto& device_info = params.get_device_info();

        const auto& weight_shape = weight_layout.get_shape();
        // When SwiGLU is fused, dispatch covers N (half of weight dim), not 2N
        bool fused_swiglu = params.has_fused_primitives();
        size_t m = fused_swiglu ? weight_shape[1] / 2 : weight_shape[1];
        size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];

        // Dispatch per (token, expert_slot) pair. Each workgroup processes one token
        // with one expert's weights, since different tokens may route to different experts.
        // z-dim = n_tokens * top_k (flat index for each token-expert pair)
        // y-dim = 1 (single token per dispatch)
        size_t n_tokens = static_cast<size_t>(rtp->n_tokens);
        size_t top_k = static_cast<size_t>(rtp->top_k);

        wgs.local = {sg_per_wg_m * get_subgroup_size(device_info.arch), sg_per_wg_n, sg_per_wg_k};
        wgs.global = {ceil_div(m, wg_tile_m), 1, n_tokens * top_k};
        wgs.global[0] *= wgs.local[0];
        wgs.global[1] *= wgs.local[1];
        wgs.global[2] *= wgs.local[2];

        ScalarDescriptor s_m{ScalarDescriptor::Types::INT32};
        s_m.v.s32 = static_cast<int32_t>(m);
        scalars.push_back(s_m);
        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = static_cast<int32_t>(k);
        scalars.push_back(s_k);
    }};
}

std::string GatherMatmulMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    return base_options + extra_options;
}

Arguments GatherMatmulMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    auto cfg = get_config(params);

    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INDICES});
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    if (cfg.has_bias) {
        args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::BIAS});
    }

    if (cfg.is_weight_quantized) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
        if (!cfg.is_weight_symmetric_quantized)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});
    }

    return args;
}

KernelData GatherMatmulMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    micro::Package bgm_gemm;
    const auto& device_info = params.get_device_info();
    try {
        init_microkernels(params, bgm_gemm, m_is_prefill);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("GatherMatmulMicroGenerator::get_kernel_data() - can't init microkernels: ", ex.what());
    }

    auto jit = get_jit_constants(params, bgm_gemm, get_config(params));

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

    // Generate microkernel shims with "gm" decorator
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = static_cast<int32_t>(get_subgroup_size(device_info.arch));
    shim_options.useTileOps = true;
    shim_options.decorator = "gm";

    kd.code->jit += generateShim(bgm_gemm, micro::HostLanguage::OpenCL_C, shim_options);
    if (bgm_gemm.grfMin > 128) {
        kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(bgm_gemm));

    uint32_t slm_size = kd.micro_kernels[0]->p.getSetting("slm_size");
    kd.params.local_memory_args.clear();
    if (slm_size > 0) {
        kd.params.local_memory_args.push_back(slm_size);
        kd.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
    }
    return kd;
}
}  // namespace ov::intel_gpu::ocl
#endif
