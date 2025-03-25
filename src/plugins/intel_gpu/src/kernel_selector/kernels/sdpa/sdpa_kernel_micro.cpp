// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "sdpa_kernel_micro.h"
#include "common_tools.h"
#include "common_types.h"
#include "jitter.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "micro_utils.hpp"
#include "tensor_type.h"

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace kernel_selector {

namespace {

size_t subgroup_size(gpu_arch arch) {
    switch (arch) {
        case gpu_arch::gen9:
        case gpu_arch::gen11:
        case gpu_arch::xe_lp:
        case gpu_arch::xe_hp:
        case gpu_arch::xe_hpg: return 8;
        case gpu_arch::xe_hpc:
        case gpu_arch::xe2:
        case gpu_arch::xe3: return 16;
        default: return 0;
    }
}

inline int64_t get_d_max(int64_t head_size) {
    for (int64_t i = 32; i <= 1024; i *= 2)
        if (head_size <= i)
            return i;
    return head_size;
}

micro::Type convert_type(Datatype t) {
    switch (t) {
        case Datatype::F32: return micro::Type::f32;
        case Datatype::F16: return micro::Type::f16;
        case Datatype::INT8: return micro::Type::s8;
        case Datatype::UINT8: return micro::Type::u8;
        default: break;
    }
    OPENVINO_THROW("Unsupported dt: ", toString(t));
}

Tensor::NDims normalize_dims(const DataTensor& qkv) {
    auto dims = qkv.GetDims(); // xyfb
    std::reverse(dims.begin(), dims.end()); // bfyx
    return dims;
}

Tensor::Dim get_num_heads(const sdpa_params& params, const DataTensor& qkv, const std::vector<int64_t>& order) {
    if (params.conf.is_paged_attention)
        return normalize_dims(qkv)[1].v / params.conf.head_size;

    return normalize_dims(qkv)[order[1]];
}

Tensor::Dim get_seq_length(const sdpa_params& params, const DataTensor& qkv, const std::vector<int64_t>& order) {
    if (params.conf.is_paged_attention)
        return Tensor::Dim(params.conf.paged_attention_aligned_seq_len);

    return normalize_dims(qkv)[order[2]];
}

struct sdpa_config_t {
    int unroll_m_kq, unroll_n_kq; // Subgroup tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs; // Subgroup tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq; // Workgroup configuration for K*Q GEMM
    int wg_m_vs, wg_n_vs; // Workgroup configuration for V*S GEMM
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

sdpa_config_t *choose_config_xehpg(int head_size, int seq, bool thin_q, bool quantized, bool is_pa) {
    if (head_size <= 32) {
        if (quantized && seq >= 128) {
            if (thin_q) return &xehpg_q_h32_2nd;
            return &xehpg_q_h32;
        }
        if (thin_q) return &xehpg_h32_2nd;
        if (seq <= 0 && is_pa) return &xehpg_h32;
        if (seq <= 32) return &xehpg_h32_s32;
        if (seq <= 64) return &xehpg_h32_s64;
        if (seq <= 256) return &xehpg_h32_s256;
        return &xehpg_h32;
    } else if (head_size <= 64) {
        if (quantized) {
            if (thin_q) return &xehpg_q_h64_2nd;
            return &xehpg_q_h64;
        }
        if (thin_q) return &xehpg_h64_2nd;
        if (seq <= 0 && is_pa) return &xehpg_h64;
        if (seq <= 64) return &xehpg_h64_s64;
        if (seq <= 128) return &xehpg_h64_s128;
        return &xehpg_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 64) return &xehpg_q_h128_s64_2nd;
                return &xehpg_q_h128_2nd;
            }
            if (seq <= 32) return &xehpg_h128_s32;
            return &xehpg_q_h128;
        }
        if (thin_q) {
            if (seq <= 256) return &xehpg_h128_s256_2nd;
            return &xehpg_h128_2nd;
        }
        if (seq <= 0 && is_pa) return &xehpg_h128;
        if (seq <= 32) return &xehpg_h128_s32;
        return &xehpg_h128;
    } else if (head_size <= 256) {
        if (thin_q) {
            if (seq <= 32) return &xehpg_h256_s32_2nd;
            if (seq <= 64) return &xehpg_h256_s64_2nd;
            return &xehpg_h256_2nd;
        }
        if (seq <= 0 && is_pa) return &xehpg_h256;
        if (seq <= 32) return &xehpg_h256_s32;
        if (seq <= 128) return &xehpg_h256_s128;
        return &xehpg_h256;
    }
    return nullptr;
}

sdpa_config_t *choose_config_xehpc(int head_size, int seq, bool thin_q, bool quantized, bool is_pa) {
    if (head_size <= 32) {
        if (thin_q) return &xehpc_h32_2nd;
        if (seq <= 0 && is_pa) return &xehpc_h32;
        if (seq <= 32) return &xehpc_h32_s32;
        return &xehpc_h32;
    } else if (head_size <= 64) {
        if (thin_q) {
            if (seq <= 64) return &xehpc_h64_s64_2nd;
            return &xehpc_h64_2nd;
        }
        if (quantized && seq >= 256) return &xehpc_q_h64;
        if (seq <= 0 && is_pa) return &xehpc_h64;
        if (seq <= 32) return &xehpc_h64_s32;
        if (seq <= 64) return &xehpc_h64_s64;
        return &xehpc_h64;
    } else if (head_size <= 128) {
        if (quantized) {
            if (thin_q) {
                if (seq <= 32) return &xehpc_q_h128_s32_2nd;
                return &xehpc_q_h128_2nd;
            }
            if (seq <= 32) return &xehpc_q_h128_s32;
            if (seq <= 64) return &xehpc_q_h128_s64;
            return &xehpc_q_h128;
        }
        if (thin_q) return &xehpc_h128_2nd;
        if (seq <= 0 && is_pa) return &xehpc_h128;
        if (seq <= 32) return &xehpc_h128_s32;
        if (seq <= 64) return &xehpc_h128_s64;
        return &xehpc_h128;
    } else if (head_size <= 256) {
        if (thin_q) return &xehpc_h256_2nd;
        if (seq <= 0 && is_pa) return &xehpc_h256;
        if (seq <= 64) return &xehpc_h256_s64;
        return &xehpc_h256;
    }
    return nullptr;
}

}  // namespace

const bool kq_common_scales = false;
const bool kq_common_zp = false;
const bool vs_common_scales = false;
const bool vs_common_zp = false;

std::mutex SDPAKernelMicro::m;

void SDPAKernelMicro::init_microkernels(const sdpa_params& params, micro::Package& gemm_kq, micro::Package& gemm_vs, bool is_prefill) const {
    // TODO: Remove once micro API is thread safe
    std::lock_guard<std::mutex> l(m);
    const auto& Q = params.inputs[0];
    const auto& K = params.inputs[1];
    const auto& V = params.inputs[2];

    auto& out = params.outputs[0];
    const auto head_size = params.conf.head_size;
    const auto d_max = get_d_max(head_size);
    const Tensor::Dim n_keys = get_seq_length(params, K, params.input1_order);
    const Tensor::Dim n_queries = get_seq_length(params, Q, params.input0_order);
    const Tensor::Dim n_values = Tensor::Dim(head_size);
    const auto batch = out.Batch().v * out.Feature().v;

    /* Retrieve pre-tuned kernel configuration */
    sdpa_config_t *config = nullptr;
    bool thin_q = (!n_queries.is_dynamic && (n_queries.v <= 16)) || !is_prefill;

    bool is_quantized = (K.GetDType() == Datatype::UINT8 || K.GetDType() == Datatype::INT8) ||
                        (V.GetDType() == Datatype::UINT8 || V.GetDType() == Datatype::INT8);

    switch (params.engineInfo.arch) {
        case gpu_arch::xe_hpg: {
            config = choose_config_xehpg(static_cast<int32_t>(head_size), static_cast<int32_t>(n_keys.v), thin_q, is_quantized, params.conf.is_paged_attention);
            break;
        }
        case gpu_arch::xe_hpc:
        case gpu_arch::xe2:
        case gpu_arch::xe3: {
            config = choose_config_xehpc(static_cast<int32_t>(head_size), static_cast<int32_t>(n_keys.v), thin_q, is_quantized, params.conf.is_paged_attention);
            break;
        }
        default: break;
    }

    OPENVINO_ASSERT(config != nullptr);

    /* Get device information */
    micro::HWInformation hw_info;
    hw_info.euCount = params.engineInfo.computeUnitsCount;
    hw_info.gmdid = params.engineInfo.ip_version;
    hw_info.systolicAvailable = params.engineInfo.supports_immad;

    /* Set up GEMMProblem structure for first GEMM: K^T * Q */
    micro::GEMMProblem problem;
    problem.Ta_ext = convert_type(K.GetDType());
    problem.Tb_ext = convert_type(Q.GetDType());

    problem.Ta = problem.Tb = micro::Type::f16;
    problem.Tc = problem.Tc_ext = micro::Type::f32;
    problem.Ts = problem.Tc;

    auto problem_kq = problem;
    problem_kq.A.layout = micro::MatrixLayout::T;

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_kq;
    opts_kq.localB = true;
    opts_kq.slmPtr = true;

    if (params.conf.is_kv_compressed && !kq_common_scales) {
        const auto scale_dt = convert_type(params.key_cache_comp_scale.GetDType());
        problem_kq.Ta_scale = scale_dt;
        problem_kq.A_scale.alignment = micro::data_type_size(scale_dt);

        problem_kq.A_scale.layout = micro::MatrixLayout::T;
        problem_kq.aScale2D = true;
    }

    if (params.conf.is_kv_compressed && params.conf.use_asymmetric_quantization) {
        const auto zp_dt = convert_type(params.key_cache_comp_zp.GetDType());
        problem_kq.Tao = zp_dt;
        problem_kq.AO.alignment = micro::data_type_size(zp_dt);
        problem_kq.AO.layout = micro::MatrixLayout::T;
        problem_kq.aoPtrDims = kq_common_zp ? 0 : 2;
        problem_kq.aOffset = micro::ABOffset::Calc;
    }

    if (params.conf.is_kv_compressed) {
        problem_kq.aqGroupM = 1;
        problem_kq.aqGroupK = (kq_common_scales || kq_common_zp) ? 1 : params.conf.head_size;
    }

    opts_kq.scaleA = params.conf.is_kv_compressed && !kq_common_scales;
    opts_kq.offsetA = params.conf.is_kv_compressed && params.conf.use_asymmetric_quantization;

    problem_kq.B.layout = micro::MatrixLayout::Pr;
    problem_kq.C.layout = micro::MatrixLayout::T;
    problem_kq.A.setAlignment(micro::alignment_for_ld(head_size * problem.Ta));
    problem_kq.B.setAlignment(64); // Q is packed in VNNI format in SLM
    problem_kq.B.crosspack = 2;
    problem_kq.B.tileR = d_max;
    problem_kq.B.tileC = static_cast<uint16_t>(subgroup_size(params.engineInfo.arch));

    /* Set up problem size information */
    micro::SizeParams sizes;
    sizes.m = static_cast<int64_t>(n_keys.v);
    sizes.n = static_cast<int64_t>(n_queries.v);
    sizes.k = static_cast<int64_t>(head_size);
    sizes.batch = static_cast<int64_t>(batch);

    /* Set up microkernel requirements */
    std::vector<micro::StrategyRequirement> reqs_kq;
    reqs_kq.push_back(micro::StrategyRequirement::UnrollM == config->unroll_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::UnrollN == config->unroll_n_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGM == config->wg_m_kq);
    reqs_kq.push_back(micro::StrategyRequirement::WGN == config->wg_n_kq);

    /* Ask microkernel provider for microkernel */
    try {
        gemm_kq = micro::select_gemm_microkernel(opts_kq, hw_info, sizes, problem_kq, reqs_kq);
    } catch (const std::runtime_error &ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create KQ sdpa_micro kernel: " << ex.what() << "\n";
        throw;
    }

    /* Set up microkernel options */
    micro::GEMMProtocol::Options opts_vs;
    opts_vs.localB = true;
    opts_vs.slmPtr = true;

    /* Update for second GEMM: V*S */
    auto problem_vs = problem;
    problem_vs.Ta_ext = convert_type(V.GetDType());
    problem_vs.A.layout = micro::MatrixLayout::N;

    if (params.conf.is_kv_compressed && !vs_common_scales) {
        auto scale_dt = convert_type(params.value_cache_comp_scale.GetDType());
        problem_vs.Ta_scale = scale_dt;
        problem_vs.A_scale.alignment = micro::data_type_size(scale_dt);
        problem_vs.A_scale.layout = micro::MatrixLayout::N;
        problem_vs.aScale2D = true;
    }

    if (params.conf.is_kv_compressed && params.conf.use_asymmetric_quantization) {
        auto zp_dt = convert_type(params.value_cache_comp_zp.GetDType());
        problem_vs.Tao = zp_dt;
        problem_vs.AO.alignment = micro::data_type_size(zp_dt);
        problem_vs.AO.layout = micro::MatrixLayout::N;
        problem_vs.aoPtrDims = vs_common_zp ? 0 : 2;
        problem_vs.aOffset = micro::ABOffset::Calc;
    }

    if (params.conf.is_kv_compressed) {
        problem_vs.aqGroupM = (vs_common_scales || vs_common_zp) ? 1 : micro::rnd_up_pow2(params.conf.head_size);
        problem_vs.aqGroupK = 1;
    }

    opts_vs.scaleA = params.conf.is_kv_compressed && !vs_common_scales;
    opts_vs.offsetA = params.conf.is_kv_compressed && params.conf.use_asymmetric_quantization;

    problem_vs.B.layout = micro::MatrixLayout::Pr;
    problem_vs.C.layout = micro::MatrixLayout::N;
    problem_vs.A.setAlignment(micro::alignment_for_ld(head_size * problem.Ta));
    problem_vs.B.setAlignment(64); // S is packed in SLM
    problem_vs.B.crosspack = 16;
    sizes.m = static_cast<int64_t>(n_values.v);
    sizes.n = gemm_kq.getSetting("wg_tile_n");
    sizes.k = gemm_kq.getSetting("wg_tile_m");

    /* Set up special kernel requirements */
    std::vector<micro::StrategyRequirement> reqs_vs;
    reqs_vs.push_back(micro::StrategyRequirement::UnrollM == config->unroll_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::UnrollN == config->unroll_n_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGM == config->wg_m_vs);
    reqs_vs.push_back(micro::StrategyRequirement::WGN == config->wg_n_vs);

    auto adjust_vs = [](micro::GEMMStrategy &strategy) {
        /* Enable dpasw */
        strategy.dpasw |= strategy.fused;
    };
    /* Ask microkernel provider for microkernel */
    try {
        gemm_vs = micro::select_gemm_microkernel(opts_vs, hw_info, sizes, problem_vs, reqs_vs, adjust_vs);
    } catch (const std::runtime_error &ex) {
        GPU_DEBUG_TRACE_DETAIL << "Can't create VS sdpa_micro kernel: " << ex.what() << "\n";
        throw;
    }
}

ParamsKey SDPAKernelMicro::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

bool SDPAKernelMicro::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        return false;

    const sdpa_params& params = static_cast<const sdpa_params&>(p);

    if (params.should_use_sdpa_opt)
        return false;

    if (params.engineInfo.arch < gpu_arch::xe_hpg || !params.engineInfo.supports_microkernels)
        return false;

    if (params.indirect_axis != -1)
        return false;

    auto Q_num_heads_dim = params.conf.is_paged_attention ? params.conf.heads_num
                                                          : get_num_heads(params, params.inputs[0], params.input0_order);
    auto K_num_heads_dim = get_num_heads(params, params.inputs[1], params.input1_order);
    auto V_num_heads_dim = get_num_heads(params, params.inputs[2], params.input2_order);

    if (params.input0_order[3] != 3 || params.input1_order[3] != 3 || params.input2_order[3] != 3)
        return false;

    if (Q_num_heads_dim.is_dynamic || K_num_heads_dim.is_dynamic || V_num_heads_dim.is_dynamic || K_num_heads_dim.v != V_num_heads_dim.v)
        return false;

    if (params.conf.head_size > 256)
        return false;

    // Do not use sdpa_micro kernel with a scalar-value mask
    const auto scale_idx = params.conf.is_paged_attention ? 4lu : 3lu;
    if (params.inputs.size() > scale_idx && !params.inputs[scale_idx].is_dynamic() && params.inputs[scale_idx].LogicalSize() == 1)
        return false;

    // Scores output is not supported
    if (params.conf.is_paged_attention && params.outputs.size() > 1)
        return false;

    if (params.conf.is_paged_attention && params.conf.paged_attention_sliding_window != 0) {
        return false;
    }

    // Alibi is not supported
    if (params.conf.is_paged_attention && params.conf.has_alibi_input)
        return false;

    return true;
}

JitConstants SDPAKernelMicro::GetJitConstants(const sdpa_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const {
    auto jit = MakeBaseParamsJitConstants(params);
    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);

    const auto& Q = prim_params.inputs[0];
    const auto& K = prim_params.inputs[1];
    const auto& V = prim_params.inputs[2];

    const auto head_size = prim_params.conf.head_size;

    auto ldq = head_size * Q.ElementSize();
    auto ldk = head_size * K.ElementSize();
    auto ldv = head_size * V.ElementSize();
    auto lda = head_size * prim_params.outputs[0].ElementSize();

    const auto d_max = get_d_max(head_size);
    const auto n_keys = get_seq_length(params, K, prim_params.input1_order);
    const auto n_queries = get_seq_length(params, Q, prim_params.input0_order);
    const auto n_values = Tensor::Dim(head_size);

    auto data_inputs = params.inputs.size();
    if (params.conf.is_paged_attention)
        data_inputs--;

    jit.AddConstant(MakeJitConstant("D_MAX", d_max));
    jit.AddConstant(MakeJitConstant("SUBGROUP_SIZE", subgroup_size(prim_params.engineInfo.arch)));
    jit.AddConstant(MakeJitConstant("INVERT_SCALE", false));
    jit.AddConstant(MakeJitConstant("SCALE_DATA_T", "half"));
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", head_size));
    jit.AddConstant(MakeJitConstant("WITH_CAUSAL_MASK", params.conf.is_causal));

    jit.AddConstant(MakeJitConstant("WITH_ATTN_MASK", data_inputs > 3));
    jit.AddConstant(MakeJitConstant("WITH_SCALE", data_inputs > 4));
    jit.AddConstant(MakeJitConstant("Q_ALIGN", micro::alignment_for_ld(ldq)));
    jit.AddConstant(MakeJitConstant("K_ALIGN", micro::alignment_for_ld(ldk)));
    jit.AddConstant(MakeJitConstant("V_ALIGN", micro::alignment_for_ld(ldv)));
    jit.AddConstant(MakeJitConstant("A_ALIGN", micro::alignment_for_ld(lda)));

    jit.AddConstant(MakeJitConstant("TRANSPOSE_K", false));
    jit.AddConstant(MakeJitConstant("IS_PAGED_ATTENTION", params.conf.is_paged_attention));
    jit.AddConstant(MakeJitConstant("KV_HEADS_NUM", params.conf.kv_heads_num));
    jit.AddConstant(MakeJitConstant("HEADS_NUM", params.conf.heads_num));

    jit.AddConstant(MakeJitConstant("QRY_DATA_T", toCLType(Q.GetDType())));
    jit.AddConstant(MakeJitConstant("KEY_DATA_T", toCLType(K.GetDType())));
    jit.AddConstant(MakeJitConstant("VAL_DATA_T", toCLType(V.GetDType())));

    if (params.conf.is_kv_compressed) {
        jit.AddConstant(MakeJitConstant("KV_COMPRESSED", 1));
        jit.AddConstant(MakeJitConstant("KEY_ATTR_SCALES_DATA_T", toCLType(params.key_cache_comp_scale.GetDType())));
        jit.AddConstant(MakeJitConstant("VAL_ATTR_SCALES_DATA_T", toCLType(params.value_cache_comp_scale.GetDType())));

        if (params.conf.use_asymmetric_quantization) {
            jit.AddConstant(MakeJitConstant("KEY_ATTR_ZP_DATA_T", toCLType(params.key_cache_comp_zp.GetDType())));
            jit.AddConstant(MakeJitConstant("VAL_ATTR_ZP_DATA_T", toCLType(params.value_cache_comp_zp.GetDType())));
        }
    }

    auto elems_per_byte = [](Datatype dt) {
        switch (dt) {
            case Datatype::UINT4:
            case Datatype::INT4:
                return 2;
            default:
                return 1;
        }
    };

    jit.AddConstant(MakeJitConstant("KEY_ELEMENTS_PER_BYTE", elems_per_byte(params.inputs[1].GetDType())));
    jit.AddConstant(MakeJitConstant("VAL_ELEMENTS_PER_BYTE", elems_per_byte(params.inputs[2].GetDType())));

    if (params.conf.is_kv_compressed) {
        int kq_scale_mask = (static_cast<int>(params.conf.is_kv_compressed) << 1) | static_cast<int>(kq_common_scales);
        int vs_scale_mask = (static_cast<int>(params.conf.is_kv_compressed) << 1) | static_cast<int>(vs_common_scales);
        jit.AddConstant(MakeJitConstant("KEY_SCALES", kq_scale_mask));
        jit.AddConstant(MakeJitConstant("VAL_SCALES", vs_scale_mask));
        jit.AddConstant(MakeJitConstant("KEY_GROUP_SIZE", params.conf.head_size));
        jit.AddConstant(MakeJitConstant("VAL_GROUP_SIZE", params.conf.head_size));

        if (params.conf.use_asymmetric_quantization) {
            int kq_zp_mask = (static_cast<int>(params.conf.use_asymmetric_quantization) << 1) | static_cast<int>(kq_common_zp);
            int vs_zp_mask = (static_cast<int>(params.conf.use_asymmetric_quantization) << 1) | static_cast<int>(vs_common_zp);
            jit.AddConstant(MakeJitConstant("KEY_ZERO_POINTS", kq_zp_mask));
            jit.AddConstant(MakeJitConstant("VAL_ZERO_POINTS", vs_zp_mask));
            jit.AddConstant(MakeJitConstant("KEY_ZP_ELEMENTS_PER_BYTE", elems_per_byte(params.key_cache_comp_zp.GetDType())));
            jit.AddConstant(MakeJitConstant("VAL_ZP_ELEMENTS_PER_BYTE", elems_per_byte(params.value_cache_comp_zp.GetDType())));
        }
    }

    int tile_k = gemm_kq.getSetting("wg_tile_m");
    int tile_q = gemm_kq.getSetting("wg_tile_n");
    int tile_v = gemm_vs.getSetting("wg_tile_m");

    bool d_full = (head_size == d_max);
    bool v_full = (head_size == tile_v);
    bool k_full = !n_keys.is_dynamic && (n_keys.v % tile_k) == 0;
    bool q_full = !n_queries.is_dynamic && (n_queries.v % tile_q) == 0;

    // WA for PA for Qwen model as it has shape with an upper bound [?, ..134213632]
    // instead of ordinary fused [?, HEAD_SIZE * HEADS_NUM], so read heads_num from config
    auto Q_num_heads_dim = params.conf.is_paged_attention ? Tensor::Dim(params.conf.heads_num)
                                                          : get_num_heads(params, params.inputs[0], params.input0_order);
    auto K_num_heads_dim = get_num_heads(params, K, params.input1_order);

    jit.AddConstant(MakeJitConstant("REMAINDER_K", !k_full));
    jit.AddConstant(MakeJitConstant("KV_GROUP_SIZE", Q_num_heads_dim.v / K_num_heads_dim.v));

    if (d_full) {
        if (ldq % 4 == 0)
            jit.AddConstant(MakeJitConstant("BLOCK_Q", 1));
        // TODO: Causes accuracy drop for static SD model. Enable back once the issue is resolved
        // if (lda % 4 == 0 && v_full)
        //     jit.AddConstant(MakeJitConstant("BLOCK_A", 1));
        jit.AddConstant(MakeJitConstant("REMAINDER_Q", !q_full));
    } else if (params.engineInfo.arch >= gpu_arch::xe_hpc) {
        auto vbytes = n_values.v * V.ElementSize();
        if (lda % 16 == 0 && vbytes % 4 == 0)
            jit.AddConstant(MakeJitConstant("BLOCK_2D_A", 1));
    }

    if (params.engineInfo.arch >= gpu_arch::xe_hpc) {
        jit.AddConstant(MakeJitConstant("PREFETCH_MASK", 1));
        jit.AddConstant(MakeJitConstant("PREFETCH_K0", 1));
        jit.AddConstant(MakeJitConstant("PREFETCH_K", 1));
        jit.AddConstant(MakeJitConstant("PREFETCH_V", 1));
        bool no_rem = d_full && v_full && k_full;
        jit.AddConstant(MakeJitConstant("PREFETCH_REMAINDER", !no_rem));
        jit.AddConstant(MakeJitConstant("PREFETCH_D_MAX", std::min<int64_t>(d_max, 64)));
    }

    auto unit_parameters = [](std::string prefix) {
        JitConstants definitions({});
        for (size_t i = 0; i < 4; i++) {
            definitions.AddConstant(MakeJitConstant(prefix + "_B" + std::to_string(i), 1));
            definitions.AddConstant(MakeJitConstant(prefix + "_SB" + std::to_string(i), 1));
        }

        return definitions;
    };

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
            definitions.AddConstant(MakeJitConstant(target_stride_definitions[i], source_stride_definitions[order[i]]));
            definitions.AddConstant(MakeJitConstant(target_size_definitions[i], source_size_definitions[order[i]]));
        }

        return definitions;
    };

    jit.Merge(convert_strides("QRY", "INPUT0", prim_params.input0_order));
    jit.Merge(convert_strides("KEY", "INPUT1", prim_params.input1_order));
    jit.Merge(convert_strides("VAL", "INPUT2", prim_params.input2_order));
    jit.Merge(convert_strides("DST", "OUTPUT", prim_params.output_order));

    jit.Merge(unit_parameters("QRY"));
    jit.Merge(unit_parameters("KEY"));
    jit.Merge(unit_parameters("VAL"));
    jit.Merge(unit_parameters("DST"));

    if (params.inputs.size() > 3) {
        jit.Merge(convert_strides("MSK", "INPUT3", {0, 1, 2, 3}));
        jit.Merge(unit_parameters("MSK"));
    }

    if (params.conf.is_kv_compressed) {
        jit.AddConstant(MakeJitConstant("KEY_SCALE", params.key_cache_comp_scale));
        jit.AddConstant(MakeJitConstant("VAL_SCALE", params.value_cache_comp_scale));

        const std::vector<int64_t> default_order = { 0, 1, 2, 3 };
        jit.Merge(convert_strides("KEY_COMP", "KEY_SCALE", default_order));
        jit.Merge(convert_strides("VAL_COMP", "VAL_SCALE", default_order));

        jit.Merge(unit_parameters("KEY_COMP"));
        jit.Merge(unit_parameters("VAL_COMP"));
    }

    return jit;
}

CommonDispatchData SDPAKernelMicro::SetDefault(const sdpa_params& params, const micro::Package& gemm_kq, const micro::Package& gemm_vs) const {
    CommonDispatchData dispatch_data;

    auto wg_tile_q = gemm_kq.getSetting("wg_tile_n");
    auto sg_per_wg = gemm_kq.getSetting("sg_per_wg_m") * gemm_kq.getSetting("sg_per_wg_n");

    dispatch_data.lws = {subgroup_size(params.engineInfo.arch), (size_t)sg_per_wg, 1};
    dispatch_data.gws = dispatch_data.lws;

    auto seq_length = get_seq_length(params, params.inputs[0], params.input0_order).v;
    auto heads_num = params.conf.is_paged_attention ? params.conf.heads_num : params.outputs[0].Feature().v;
    auto batch_size = params.conf.is_paged_attention ? 1 : params.outputs[0].Batch().v;

    dispatch_data.gws[0] *= CeilDiv(seq_length, wg_tile_q);
    dispatch_data.gws[1] *= heads_num;
    dispatch_data.gws[2] *= batch_size;

    return dispatch_data;
}

clKernelData SDPAKernelMicro::get_kernel_data(const sdpa_params& params, bool is_prefill) const {
    auto name = kernelName + (is_prefill ? "_prefill" : "_generate");
    if (params.conf.is_paged_attention)
        name = "pa_" + name;

    std::vector<micro::Package> gemms(2); // KQ and VS
    init_microkernels(params, gemms[kq_id], gemms[vs_id], is_prefill);
    auto dispatch_data = SetDefault(params, gemms[kq_id], gemms[vs_id]);
    auto entry_point = GetEntryPoint(name, params.layerID, params);
    auto jit = CreateJit(name, GetJitConstants(params, gemms[kq_id], gemms[vs_id]), entry_point);
    clKernelData kernel;

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params), 1, params.is_shape_agnostic);

    kernel.params.arguments.clear();
    if (params.is_shape_agnostic )
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1}); // K
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0}); // Q
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2}); // V
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0}); // A


    if (params.conf.is_paged_attention) {
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3}); // subsequence_begins
        if (params.inputs.size() >= 5)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4}); // scale

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3}); // paged attention helper buffer
    } else {
        if (params.inputs.size() >= 4)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3}); // mask
        if (params.inputs.size() >= 5)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 4}); // Scale

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0}); // D
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1}); // K
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 2}); // Q
    }

    if (params.conf.is_kv_compressed) {
        uint32_t input_idx = static_cast<uint32_t>(params.inputs.size());
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 0});     // K scales
        if (params.conf.use_asymmetric_quantization)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 2}); // K zp

        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 1});     // V scales
        if (params.conf.use_asymmetric_quantization)
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, input_idx + 3}); // V zp
    }

    const auto& Q = params.inputs[0];
    const auto& K = params.inputs[1];

    const auto n_queries = get_seq_length(params, Q, params.input0_order);
    const auto n_keys = get_seq_length(params, K, params.input1_order);

    auto head_size = params.conf.head_size;

    ScalarDescriptor s_d;
    s_d.t = ScalarDescriptor::Types::INT32;
    s_d.v.s32 = static_cast<uint32_t>(head_size);

    ScalarDescriptor s_k;
    s_k.t = ScalarDescriptor::Types::INT32;
    s_k.v.s32 = static_cast<uint32_t>(n_keys.v);

    ScalarDescriptor s_q;
    s_q.t = ScalarDescriptor::Types::INT32;
    s_q.v.s32 = static_cast<uint32_t>(n_queries.v);

    kernel.params.scalars.push_back(s_d);
    kernel.params.scalars.push_back(s_k);
    kernel.params.scalars.push_back(s_q);

    /* Generate microkernel shims */
    micro::ShimOptions shim_options;
    shim_options.subgroupSize = static_cast<int32_t>(subgroup_size(params.engineInfo.arch));
    shim_options.useTileOps = true;
    shim_options.decorator = "kq";

    kernel.code.kernelString->jit += generateShim(gemms[kq_id], micro::HostLanguage::OpenCL_C, shim_options);

    shim_options.microkernelID++;
    shim_options.decorator = "vs";
    kernel.code.kernelString->jit += generateShim(gemms[vs_id], micro::HostLanguage::OpenCL_C, shim_options);

    if (gemms[kq_id].grfMin > 128 || gemms[vs_id].grfMin > 128)
        kernel.code.kernelString->options += " -cl-intel-256-GRF-per-thread";

    std::string extra_options = " -Dcl_intel_dot_accumulate";
    extra_options += " -Dcl_intel_global_float_atomic";
    extra_options += " -Dcl_intel_subgroup_matrix_multiply_accumulate";
    extra_options += " -Dcl_intel_subgroup_split_matrix_multiply_accumulate";
    kernel.code.kernelString->options += extra_options;

    kernel.code.kernelString->batch_compilation = false;
    kernel.code.kernelString->has_microkernels = true;

    for (auto& p : gemms) {
        kernel.micro_kernels.push_back(std::make_shared<micro::MicroKernelPackage>(p));
    }

    return kernel;
}

KernelsData SDPAKernelMicro::GetKernelsData(const Params& params) const {
    const size_t num_kernels = params.is_shape_agnostic ? 2 : 1;
    KernelData kd = KernelData::Default<sdpa_params>(params, num_kernels);
    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);

    if (!Validate(params)) {
        return {};
    }

    for (size_t i = 0; i < num_kernels; i++) {
        kd.kernels[i] = get_kernel_data(prim_params, i == prefill_id);
    }

    GetUpdateDispatchDataFunc(kd);

    return { kd };
}

void SDPAKernelMicro::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);
        const auto& Q = prim_params.inputs[0];
        const auto& K = prim_params.inputs[1];

        const auto n_queries = get_seq_length(prim_params, Q, prim_params.input0_order);
        const auto n_keys = get_seq_length(prim_params, K, prim_params.input1_order);

        auto head_size = prim_params.conf.head_size;

        ScalarDescriptor s_d;
        s_d.t = ScalarDescriptor::Types::INT32;
        s_d.v.s32 = static_cast<uint32_t>(head_size);

        ScalarDescriptor s_k;
        s_k.t = ScalarDescriptor::Types::INT32;
        s_k.v.s32 = static_cast<uint32_t>(n_keys.v);

        ScalarDescriptor s_q;
        s_q.t = ScalarDescriptor::Types::INT32;
        s_q.v.s32 = static_cast<uint32_t>(n_queries.v);

        // TODO: Currently 2nd token version works slower than prefill version
        const bool is_prefill = true;//n_queries.v > 1;

        OPENVINO_ASSERT(kernel_data.kernels.size() == 2, "[GPU] Invalid kernels size for update dispatch data func");

        size_t target_kernel = is_prefill ? prefill_id : generate_id;

        kernel_data.kernels[prefill_id].skip_execution = true;
        kernel_data.kernels[generate_id].skip_execution = true;

        const auto& gemms = kernel_data.kernels[target_kernel].micro_kernels;
        auto dispatchData = SetDefault(prim_params, gemms[kq_id]->p, gemms[vs_id]->p);
        kernel_data.kernels[target_kernel].params.workGroups.global = dispatchData.gws;
        kernel_data.kernels[target_kernel].params.workGroups.local = dispatchData.lws;
        kernel_data.kernels[target_kernel].skip_execution = KernelData::SkipKernelExecution(prim_params);

        kernel_data.kernels[target_kernel].params.scalars.clear();
        kernel_data.kernels[target_kernel].params.scalars.push_back(s_d);
        kernel_data.kernels[target_kernel].params.scalars.push_back(s_k);
        kernel_data.kernels[target_kernel].params.scalars.push_back(s_q);

        if (prim_params.conf.is_paged_attention) {
            const auto indexes_dt = Datatype::INT32;
            const auto wg_tile_q = GetTileQSize(kernel_data);
            const auto target_seq_len = std::max(prim_params.conf.paged_attention_aligned_seq_len, static_cast<int64_t>(1));
            const auto indexes_buf_size = CeilDiv(target_seq_len, wg_tile_q) * BytesPerElement(indexes_dt) * 2;

            kernel_data.internalBuffers.clear();
            kernel_data.internalBufferDataType = indexes_dt;
            kernel_data.internalBuffers.emplace_back(indexes_buf_size, true);
        }
    };
}

KernelsPriority SDPAKernelMicro::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

size_t SDPAKernelMicro::GetTileQSize(const KernelData& kernel_data) {
    const bool is_prefill = true;//n_queries.v > 1;

    OPENVINO_ASSERT(kernel_data.kernels.size() > 0, "[GPU] Invalid kernels size for update dispatch data func, got ", kernel_data.kernels.size());
    OPENVINO_ASSERT(kernel_data.kernels[prefill_id].micro_kernels.size() > 0, "[GPU] Invalid kernels passed to GetTileQSize() function");

    size_t target_kernel = is_prefill ? prefill_id : generate_id;
    const auto& gemms = kernel_data.kernels[target_kernel].micro_kernels;
    const auto wg_tile_q = gemms[kq_id]->p.getSetting("wg_tile_n");

    return wg_tile_q;
}

}  // namespace kernel_selector

#endif // ENABLE_ONEDNN_FOR_GPU
