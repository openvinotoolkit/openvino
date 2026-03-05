// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "moe_gemm_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "moe_gemm_inst.h"
#include "../utils/kernel_generator.hpp"
#include "gemmstone/kernel_selector.hpp"

// clang-format on
namespace ov::intel_gpu::ocl {

namespace {
void entryObserver(const gemmstone::kcatalog::Entry* entry, double score, gemmstone::EvaluateAuxOutput aux) {
    GPU_DEBUG_TRACE_DETAIL << "consider strategy: " << entry->str() << ", score: " << score << "\n";
};
}  // anonymous namespace

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

JitConstants MoEGemmMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm, const moe_config& cfg) const {
    const auto& device_info = params.get_device_info();
    auto jit = make_base_jit_constants(params);
    jit.make("SUBGROUP_SIZE", get_subgroup_size(device_info.arch));
    std::vector<moe_gemm::MoEGemmInputIdx> input_ids = {moe_gemm::MoEGemmInputIdx::INPUT,
                                                        moe_gemm::MoEGemmInputIdx::WEIGHT,
                                                        moe_gemm::MoEGemmInputIdx::EXPERTS_IDS,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT,
                                                        moe_gemm::MoEGemmInputIdx::INPUT_TOKENS_LENS};
    bool has_bias = cfg.has_bias;
    bool is_u4_i4 = (params.input_layouts[1].data_type == data_types::u4 || params.input_layouts[1].data_type == data_types::i4);
    auto weight_idx = moe_gemm::MoEGemmInputIdx::WEIGHT;
    auto bias_idx = moe_gemm::MoEGemmInputIdx::BIAS;
    auto scale_idx = cfg.weight_scale_idx;
    auto zp_idx = cfg.weight_zp_idx;
    const auto& weight_shape = params.input_layouts[weight_idx].get_shape();
    if (cfg.is_weight_quantized) {
        const auto& scale_shape = params.input_layouts[scale_idx].get_shape();
        const auto& bias_shape = params.input_layouts[bias_idx].get_shape();
        if (has_bias) {
            jit.make("BIAS_DT", to_ocl_type(data_types::f16));
            jit.make("BIAS_STRIDE", bias_shape[1] * bias_shape[2]);
        }
        input_ids.push_back((moe_gemm::MoEGemmInputIdx)(static_cast<int32_t>(scale_idx)));
        if (!cfg.is_weight_symmetric_quantized)
            input_ids.push_back((moe_gemm::MoEGemmInputIdx)(static_cast<int32_t>(zp_idx)));

        jit.make("WEIGHT_SCALE_DT", to_ocl_type(data_types::f16));
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
        if (!cfg.is_weight_symmetric_quantized)
            jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::f16));
    } else {
        jit.make("EXPERT_STRIDE", params.input_layouts[1].get_shape()[1] * params.input_layouts[1].get_shape()[2]);
    }

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;
    for (size_t i = 0; i < input_ids.size(); i++) {
        const size_t tensor_id = input_ids[i];
        jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
    }
    jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
    jit.make("INPUT_STRIDE",
             params.input_layouts[1].get_shape().size() == 4 ? params.input_layouts[1].get_shape()[2] * params.input_layouts[1].get_shape()[3]
                                                             : params.input_layouts[1].get_shape()[2]);
    jit.make("OUTPUT_STRIDE", params.input_layouts[1].get_shape()[1]);
    if (!m_is_prefill)
        jit.make("IS_GENERATE", 1);
    if (cfg.has_batch_dim) {
        jit.make("INPUT_SEQ_LEN", "INPUT0_FEATURE_NUM");
    } else {
        jit.make("INPUT_SEQ_LEN", "INPUT0_BATCH_NUM");
    }
    auto slm_size = moe_gemm.getSetting("slm_size");
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

std::mutex MoEGemmMicroGenerator::mtx;
void MoEGemmMicroGenerator::init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe, bool is_prefill) noexcept {
    // TODO: Remove once micro API is thread safe
    std::lock_guard<std::mutex> l(mtx);
    auto moe_cfg = get_moe_cfg(params);
    const auto& device_info = params.get_device_info();
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;
    const auto& weight_shape = params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT).get_shape();
    size_t m = weight_shape[1];
    size_t n = is_prefill ? 32 : 8;
    size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
    GPU_DEBUG_TRACE_DETAIL << "init_microkernels for " << (is_prefill ? "prefill" : "generate") << " : Seq_len:" << n << " Ofm:" << m << " K:" << k << "\n";

    micro::GEMMProblem problem_moe;
    micro::GEMMProtocol::Options opts_moe;
    opts_moe.slmPtr = true;
    opts_moe.kParallelLocal = !is_prefill;
    enum class MICRO_DIMENSIONALITY { NONE = -1, SCALAR = 0, VECTOR = 1, MATRIX = 2 };

    if (moe_cfg.is_weight_quantized) {
        problem_moe.Ta = micro::Type::f16;
        problem_moe.Ta_ext = convert_type(params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT).data_type);
        problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta_ext));

        problem_moe.Ta_scale = convert_type(params.get_input_layout(moe_cfg.weight_scale_idx).data_type);  // zp dt
        problem_moe.A_scale.setAlignment(2);
        problem_moe.A_scale.layout = micro::MatrixLayout::T;
        problem_moe.asPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);

        problem_moe.aqGroupM = 1;
        problem_moe.aqGroupK =
            (static_cast<int32_t>(moe_cfg.weight_group_size) == -1) ? static_cast<int32_t>(k) : static_cast<int32_t>(moe_cfg.weight_group_size);

        opts_moe.scaleA = true;
        if (!moe_cfg.is_weight_symmetric_quantized) {
            const auto& zp_layout = params.get_input_layout(moe_cfg.weight_zp_idx);
            const auto zp_dt = convert_type(zp_layout.data_type);
            problem_moe.Tao = zp_dt;
            problem_moe.AO.setAlignment(zp_dt == gemmstone::Type::u4 ? 1 : static_cast<int32_t>(zp_dt.size()));
            problem_moe.AO.layout = micro::MatrixLayout::T;
            problem_moe.aoPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);
            // Calculate A/B row/column sums in kernel.
            problem_moe.aOffset = micro::ABOffset::Calc;
            opts_moe.offsetA = true;
        }
    } else {
        problem_moe.Ta = problem_moe.Ta_ext = convert_type(params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT).data_type);
        problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta));
    }

    problem_moe.Tb = problem_moe.Tb_ext = micro::Type::f16;
    problem_moe.Tc = micro::Type::f32;
    problem_moe.Tc_ext = micro::Type::f32;
    problem_moe.Ts = problem_moe.Tc;
    problem_moe.A.layout = micro::MatrixLayout::T;
    problem_moe.B.layout = micro::MatrixLayout::N;
    problem_moe.C.layout = micro::MatrixLayout::N;
    problem_moe.B.setAlignment(micro::alignment_for_ld(k * problem_moe.Tb));
    problem_moe.C.setAlignment(static_cast<int32_t>(problem_moe.Tc.size()));

    /* Set up problem_moe size information */
    micro::SizeParams sizes;
    sizes.n = static_cast<int32_t>(n);
    sizes.m = static_cast<int32_t>(m);
    sizes.k = static_cast<int32_t>(k);
    sizes.batch = static_cast<int32_t>(1);

    GPU_DEBUG_TRACE_DETAIL << "problem_moe:" << problem_moe.toString() << "\n";
    GPU_DEBUG_TRACE_DETAIL << "sizes to select gemm : m : " << m << " n : " << n << " k : " << k << std::endl;
    try {
        /* Ask microkernel provider for microkernel */
        gemmstone::SelectionObserver observer = entryObserver;
        gemm_moe = micro::select_gemm_microkernel(opts_moe, hw_info, sizes, problem_moe, &observer);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("Can't create moe micro kernel: ", ex.what());
    }
}
DispatchDataFunc MoEGemmMicroGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<MoEGemmRuntimeParams*>(rt_params);
        const auto& desc = params.typed_desc<moe_gemm>();
        const auto& device_info = params.get_device_info();
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_per_wg_k = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_k"));
        auto wg_tile_m = gemm_p.getSetting("wg_tile_m");
        auto wg_tile_n = gemm_p.getSetting("wg_tile_n");

        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(3);

        auto input_layout = params.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT);
        auto experts_weight_layout = params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
        auto output_layout = params.get_output_layout();

        size_t n = desc->has_batch_dim ? input_layout.get_shape()[1] : input_layout.get_shape()[0];
        const auto& experts_weight_shape = experts_weight_layout.get_shape();
        size_t m = experts_weight_shape[1];
        size_t k = experts_weight_shape.size() == 4 ? experts_weight_shape[2] * experts_weight_shape[3] : experts_weight_shape[2];
        wgs.local = {sg_per_wg_m * get_subgroup_size(device_info.arch), sg_per_wg_n, sg_per_wg_k};
        wgs.global = {ceil_div(m, wg_tile_m), ceil_div(n, wg_tile_n), static_cast<size_t>(rtp->num_actually_used_experts)};
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

std::string MoEGemmMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    return base_options + extra_options;
}

Arguments MoEGemmMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    auto cfg = get_moe_cfg(params);
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::EXPERTS_IDS});
    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT});

    args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::INPUT_TOKENS_LENS});

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // m
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k

    if (cfg.has_bias) {
        args.push_back({ArgumentDescriptor::Types::INPUT, moe_gemm::MoEGemmInputIdx::BIAS});
    }

    if (cfg.is_weight_quantized) {
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_scale_idx)});
        if (!cfg.is_weight_symmetric_quantized)
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(cfg.weight_zp_idx)});
    }

    return args;
}

KernelData MoEGemmMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    micro::Package moe_gemm;
    const auto& device_info = params.get_device_info();
    try {
        init_microkernels(params, moe_gemm, m_is_prefill);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("MoEGemmMicroGenerator::get_kernel_data() - can't init microkernels: ", ex.what());
    }

    auto jit = get_jit_constants(params, moe_gemm, get_moe_cfg(params));

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
    shim_options.decorator = "moe";

    kd.code->jit += generateShim(moe_gemm, micro::HostLanguage::OpenCL_C, shim_options);
    if (moe_gemm.grfMin > 128) {
        kd.code->options += " -cl-intel-256-GRF-per-thread";
    }

    kd.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(moe_gemm));

    // Micro kernel is using slm implicitly inside the kernel.
    // Therefore the slm should be allocated.
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
