// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU
// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "moe_3gemm_gen_micro.hpp"

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

// clang-format on
namespace ov::intel_gpu::ocl {

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
        return 8;
    }
}

JitConstants MoE3GemmMicroGenerator::get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm, const moe_3gemm_config& cfg) const {
    const auto& device_info = params.get_device_info();
    const size_t subgroup_size = get_subgroup_size(device_info.arch);
    const auto& weight_layout = params.input_layouts[m_wei_idx];
    const auto& scale_layout = params.input_layouts[m_scale_idx];
    const auto& zp_layout = params.input_layouts[m_zp_idx];

    // Internal generator of JIT constants, require intermediate buffers and part of primitive's inputs.
    // JitConstants jit = make_base_jit_constants(params);
    JitConstants jit;
    auto entry_point = get_entry_point(params);
    jit.add(make_jit_constant("KERNEL(name)", "__kernel void " + entry_point));
    jit.add(make_jit_constant("KERNEL_ID", entry_point));
    jit.make("OPTIONAL_SHAPE_INFO_ARG", "");
    jit.make("OPTIONAL_SHAPE_INFO_TENSOR", "");

    jit.make("SUBGROUP_SIZE", subgroup_size);
    jit.make("OUTPUT_TYPE", to_ocl_type(data_types::f16));  // output
    jit.make("INPUT0_TYPE", to_ocl_type(data_types::f16));  // input: f16

    GPU_DEBUG_TRACE_DETAIL << "\t m_wei_idx: " << m_wei_idx << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "\t m_wei_idx.get_shape(): " << weight_layout.to_short_string() << std::endl;
    const auto& weight_shape = weight_layout.get_shape();
    // weight layout: u4/u8:bfyx:4x3072x8x128:nopad
    size_t expert_stride = weight_shape.size() == 4 ? (weight_shape[1] * weight_shape[2] * weight_shape[3]) : (weight_shape[1] * weight_shape[2]);
    if (weight_layout.data_type == ov::element::u4 || weight_layout.data_type == ov::element::i4) {
        jit.make("INPUT1_TYPE", to_ocl_type(data_types::u8));  // weight: u4/i4
        jit.make("WEIGHT_COMPRESSED_INT4", 1);
        jit.make("EXPERT_STRIDE", expert_stride / 2);
        GPU_DEBUG_TRACE_DETAIL << "\t expert_stride: " << expert_stride / 2 << std::endl;
    } else {
        jit.make("INPUT1_TYPE", to_ocl_type(weight_layout.data_type));  // weight type
        jit.make("WEIGHT_COMPRESSED_INT4", 0);
        ov::element::Type dt = weight_layout.data_type;
        jit.make("EXPERT_STRIDE", expert_stride * dt.size());
        GPU_DEBUG_TRACE_DETAIL << "\t expert_stride: " << expert_stride * dt.size() << std::endl;
    }
    jit.make("INPUT2_TYPE", to_ocl_type(data_types::i32));             // experts_ids: i32
    jit.make("INPUT3_TYPE", to_ocl_type(data_types::i32));             // input_offset_per_expert: i32
    jit.make("INPUT4_TYPE", to_ocl_type(data_types::i32));             // n_array: i32
    jit.make("WEIGHT_SCALE_DT", to_ocl_type(scale_layout.data_type));  // scale: f16

    if (zp_layout.data_type == ov::element::u4 || zp_layout.data_type == ov::element::i4) {
        jit.make("WEIGHT_ZP_DT", to_ocl_type(data_types::u8));  // zp: u4/i4
        jit.make("WEIGHT_COMPRESSED_ZP_INT4", 1);
    } else {
        jit.make("WEIGHT_ZP_DT", to_ocl_type(zp_layout.data_type));  // zp type
        jit.make("WEIGHT_COMPRESSED_ZP_INT4", 0);
    }

    jit.make("IS_GENERATE", 0);    // only for prefill
    jit.make("INPUT_SEQ_LEN", 4);  // prefill not use it
    jit.make("SCALE_ZP_NO_TRANSPOSE", 1);

    GPU_DEBUG_TRACE_DETAIL << "\t m_scale_idx: " << m_scale_idx << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "\t m_scale_idx.get_shape(): " << scale_layout.to_short_string() << std::endl;
    if (cfg.weight_group_size > 0) {
        const auto scale_shape = scale_layout.get_shape();
        jit.make("NUM_GROUPS", scale_shape[1]);
        GPU_DEBUG_TRACE_DETAIL << "\t NUM_GROUPS: " << scale_shape[1] << std::endl;
    } else {
        jit.make("NUM_GROUPS", 1);
        GPU_DEBUG_TRACE_DETAIL << "\t NUM_GROUPS: 1" << std::endl;
    }

    auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
    switch (m_type) {
    case MoE3GemmMicroKernelType::MLP_GATE:
    case MoE3GemmMicroKernelType::MLP_UP:
        // f16:bfyx:[?,2048]:nopad
        jit.make("INPUT_STRIDE", desc->_config.hidden_size);
        jit.make("OUTPUT_STRIDE", desc->_config.inter_size);
        break;
    case MoE3GemmMicroKernelType::MLP_DOWN:
        jit.make("INPUT_STRIDE", desc->_config.inter_size);
        jit.make("OUTPUT_STRIDE", desc->_config.hidden_size);
        break;
    default:
        OPENVINO_THROW("Unsupported MoE3GemmMicroKernelType");
        break;
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

std::mutex MoE3GemmMicroGenerator::mtx;
std::unordered_map<MoE3GemmMicroGenerator::GemmCacheKey, micro::Package, MoE3GemmMicroGenerator::GemmCacheKeyHash> MoE3GemmMicroGenerator::s_gemm_cache;
void MoE3GemmMicroGenerator::init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe, MoE3GemmMicroKernelType type) noexcept {
    std::lock_guard<std::mutex> l(mtx);

    int wei_idx, scale_idx, zp_idx;
    auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
    size_t group_size = desc->_config.group_size;
    switch (type) {
    case MoE3GemmMicroKernelType::MLP_GATE:
        wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_0);
        scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_0);
        zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_0);
        if (group_size == std::numeric_limits<size_t>::max()) {
            group_size = desc->_config.hidden_size;
        }
        break;
    case MoE3GemmMicroKernelType::MLP_UP:
        wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_1);
        scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_1);
        zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_1);
        if (group_size == std::numeric_limits<size_t>::max()) {
            group_size = desc->_config.hidden_size;
        }
        break;
    case MoE3GemmMicroKernelType::MLP_DOWN:
        wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_2);
        scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_2);
        zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_2);
        if (group_size == std::numeric_limits<size_t>::max()) {
            group_size = desc->_config.inter_size;
        }
        break;
    default:
        OPENVINO_THROW("Unsupported MoE3GemmMicroKernelType");
        break;
    }

    const auto& weight_layout = params.get_input_layout(wei_idx);
    const auto& scale_layout = params.get_input_layout(scale_idx);
    const auto& zp_layout = params.get_input_layout(zp_idx);

    MoE3GemmMicroGenerator::GemmCacheKey key;
    key.weight_shape = weight_layout.get_shape();
    key.weight_dt = weight_layout.data_type;
    key.scale_shape = scale_layout.get_shape();
    key.scale_dt = scale_layout.data_type;
    key.zp_shape = zp_layout.get_shape();
    key.zp_dt = zp_layout.data_type;

    auto it = s_gemm_cache.find(key);
    if (it != s_gemm_cache.end()) {
        gemm_moe = it->second;
        return;
    }

    const auto& device_info = params.get_device_info();
    micro::HWInformation hw_info;
    hw_info.euCount = device_info.execution_units_count;
    hw_info.gmdid = device_info.ip_version;
    hw_info.systolicAvailable = device_info.supports_immad;

    // weight layout example: u4:bfyx:4x3072x8x128:nopad
    const auto& weight_shape = params.get_input_layout(wei_idx).get_shape();
    const bool is_prefill = true;
    size_t m = weight_shape[1];
    size_t n = is_prefill ? 32 : 8;
    size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];

    GPU_DEBUG_TRACE_DETAIL << "MoE3GemmMicroGenerator::init_microkernels: " << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "\t m = " << m << ", n = " << n << ", k = " << k << std::endl;
    GPU_DEBUG_TRACE_DETAIL << "\t weight group size: " << group_size << "\n";

    micro::GEMMProblem problem_moe;
    micro::GEMMProtocol::Options opts_moe;
    opts_moe.slmPtr = true;
    enum class MICRO_DIMENSIONALITY { NONE = -1, SCALAR = 0, VECTOR = 1, MATRIX = 2 };

    const bool is_weight_quantized = true;
    if (is_weight_quantized) {
        problem_moe.Ta = micro::Type::f16;
        problem_moe.Ta_ext = convert_type(params.get_input_layout(wei_idx).data_type);
        problem_moe.A.setAlignment(micro::alignment_for_ld(k * problem_moe.Ta_ext));

        // scale layout example: f16:bfyx:4x8x3072:nopad
        const auto& scale_layout = params.get_input_layout(scale_idx);
        problem_moe.Ta_scale = convert_type(scale_layout.data_type);
        problem_moe.A_scale.setAlignment(2);
        problem_moe.A_scale.layout = micro::MatrixLayout::N;
        problem_moe.asPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);

        problem_moe.aqGroupM = 1;
        problem_moe.aqGroupK = static_cast<int>(group_size);

        opts_moe.scaleA = true;
        const bool is_weight_symmetric_quantized = false;
        if (!is_weight_symmetric_quantized) {
            // zp layout example: u4:bfyx:4x8x3072:nopad
            const auto& zp_layout = params.get_input_layout(zp_idx);
            const auto zp_dt = convert_type(zp_layout.data_type);
            problem_moe.Tao = zp_dt;
            problem_moe.AO.setAlignment(zp_dt == micro::Type::u4 ? 1 : static_cast<int32_t>(zp_dt.size()));
            problem_moe.AO.layout = micro::MatrixLayout::N;
            problem_moe.aoPtrDims = static_cast<int>(MICRO_DIMENSIONALITY::MATRIX);
            // Calculate A/B row/column sums in kernel.
            problem_moe.aOffset = micro::ABOffset::Calc;
            opts_moe.offsetA = true;
        }
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
        micro::Package pkg = micro::select_gemm_microkernel(opts_moe, hw_info, sizes, problem_moe);
        s_gemm_cache.emplace(key, pkg);
        gemm_moe = std::move(pkg);
        GPU_DEBUG_TRACE_DETAIL << "MoE3GemmMicroGenerator::init_microkernels: create and cache new micro kernel" << std::endl;
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("Can't create moe micro kernel: ", ex.what());
    }
}
DispatchDataFunc MoE3GemmMicroGenerator::get_dispatch_data_func() const {
    const auto wei_idx = this->m_wei_idx;
    return DispatchDataFunc{[wei_idx](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());

        auto* rtp = static_cast<MoEGemmRuntimeParams*>(rt_params);
        const auto& device_info = params.get_device_info();
        const auto& gemm_p = kd.micro_kernels[0]->p;
        auto sg_per_wg_n = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_n"));
        auto sg_per_wg_m = static_cast<size_t>(gemm_p.getSetting("sg_per_wg_m"));
        auto sg_tile_m = gemm_p.getSetting("sg_tile_m");
        auto sg_tile_n = gemm_p.getSetting("sg_tile_n");

        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        scalars.reserve(2);

        auto input_layout = params.get_input_layout(0);
        auto experts_weight_layout = params.get_input_layout(wei_idx);

        GPU_DEBUG_TRACE_DETAIL << "\t input_layout: " << input_layout.to_short_string() << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "\t wei_idx = " << wei_idx << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "\t experts_weight_layout: " << experts_weight_layout.to_short_string() << std::endl;

        // has_batch_dim indicates whether the input tensor has batch dimension
        size_t n = input_layout.get_shape()[0];
        switch (input_layout.get_shape().size()) {
        case 2:
            n = input_layout.get_shape()[0];
            break;
        case 3:
        case 4:
            n = input_layout.get_shape()[0] * input_layout.get_shape()[1];
            break;
        default:
            OPENVINO_THROW("Unsupported input tensor shape size: ", input_layout.get_shape().size());
        }

        auto cur_moe = params.typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        n = n * config.top_k;
        GPU_DEBUG_TRACE_DETAIL << "\t n = " << n << std::endl;

        const auto& experts_weight_shape = experts_weight_layout.get_shape();
        const size_t subgroup_size = get_subgroup_size(device_info.arch);
        size_t m = experts_weight_shape[1];
        size_t k = experts_weight_shape.size() == 4 ? experts_weight_shape[2] * experts_weight_shape[3] : experts_weight_shape[2];
        wgs.local = {sg_per_wg_m * subgroup_size, sg_per_wg_n, 1};
        wgs.global = {align_to(ceil_div(m, sg_tile_m), sg_per_wg_m) * subgroup_size,
                      align_to(ceil_div(n, sg_tile_n), sg_per_wg_n),
                      static_cast<size_t>(rtp->num_actually_used_experts)};
        ScalarDescriptor s_m{ScalarDescriptor::Types::INT32};
        s_m.v.s32 = static_cast<int32_t>(m);
        scalars.push_back(s_m);
        ScalarDescriptor s_k{ScalarDescriptor::Types::INT32};
        s_k.v.s32 = static_cast<int32_t>(k);
        scalars.push_back(s_k);

        GPU_DEBUG_TRACE_DETAIL << "\t m = " << m << ", k = " << k << std::endl;
    }};
}

std::string MoE3GemmMicroGenerator::get_build_options(const kernel_impl_params& params) const {
    auto base_options = KernelGenerator::get_build_options(params);
    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    return base_options + extra_options;
}

Arguments MoE3GemmMicroGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    auto desc = params.typed_desc<moe_3gemm_fused_compressed>();

    switch (m_type) {
    case MoE3GemmMicroKernelType::MLP_GATE:
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_GATE_UP_INPUT});  // gather input tensor
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::WEIGHT_0)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_GATE_OUTPUT});                     // gate output
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS});            // experts_ids
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT});   // input_offset_per_expert
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT});  // n_array - token len
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});                                                            // m
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});                                                            // k
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::SCALE_0)});                 // scale
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::ZP_0)});                    // zp
        break;
    case MoE3GemmMicroKernelType::MLP_UP:
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_GATE_UP_INPUT});  // gather input tensor
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::WEIGHT_1)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_UP_OUTPUT});                       // up output
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS});            // experts_ids
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT});   // input_offset_per_expert
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT});  // n_array - token len
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});                                                            // m
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});                                                            // k
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::SCALE_1)});                 // scale
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::ZP_1)});                    // zp
        break;
    case MoE3GemmMicroKernelType::MLP_DOWN:
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_GATE_OUTPUT});  // intermediate_mem[6]
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::WEIGHT_2)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_DOWN_OUTPUT});                     // down output
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS});            // experts_ids
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT});   // input_offset_per_expert
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT});  // n_array - token len
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});                                                            // m
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});                                                            // k
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::SCALE_2)});                 // scale
        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<int>(MOE3GemmInputIndex::ZP_2)});                    // zp
        break;
    default:
        OPENVINO_THROW("Unsupported MoE3GemmMicroKernelType");
        break;
    }

    return args;
}

KernelData MoE3GemmMicroGenerator::get_kernel_data(const kernel_impl_params& params) const {
    micro::Package moe_gemm;
    const auto& device_info = params.get_device_info();
    try {
        init_microkernels(params, moe_gemm, m_type);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("MoE3GemmMicroGenerator::get_kernel_data() - can't init microkernels: ", ex.what());
    }

    auto jit = get_jit_constants(params, moe_gemm, get_moe_3gemm_cfg(params));

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = kernel_language::OCLC_V2;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = false;
    kd.code->has_microkernels = true;

    try {
        GPU_DEBUG_TRACE_DETAIL << "\t get_kernel_name(): " << get_kernel_name() << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "\t kd.code->entry_point: " << kd.code->entry_point << std::endl;
        kd.code->str = build_code(get_kernel_name(), jit, kd.code->entry_point);
    } catch (const std::runtime_error& ex) {
        OPENVINO_THROW("MoE3GemmMicroGenerator::get_kernel_data() - can't build code: ", ex.what());
    }

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

    GPU_DEBUG_TRACE_DETAIL << "MoE3GemmMicroGenerator::get_kernel_data() completed\n";
    return kd;
}
}  // namespace ov::intel_gpu::ocl
#endif
