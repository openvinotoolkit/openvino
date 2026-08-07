// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "moe_3gemm_gen_micro.hpp"
#include "moe_otd_runtime.hpp"
#include "moe_3gemm_swiglu_opt.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "lru_cache.hpp"
#include "expert_weight_providers.hpp"
// clang-format on

using ov::intel_gpu::ocl::moe::IExpertWeightProvider;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;

#define DEBUG_MOE_LOG 0

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <algorithm>
#    include <chrono>
#    include <cstdint>
#    include <fstream>
#    include <initializer_list>
#    include <iostream>
#    include <limits>
#    include <mutex>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_ocl.hpp>
#    include <sstream>
#    include <string_view>
#    include <thread>
#    include <tuple>
#    include <utility>

#    include "../primitive_ocl_base.hpp"
#    include "../utils/kernel_generator.hpp"
#    include "common_utils/jitter.hpp"
#    include "impls/onednn/grouped_matmul_helper.hpp"
#    include "intel_gpu/graph/kernel_impl_params.hpp"
#    include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#    include "intel_gpu/runtime/stream.hpp"
#    include "moe_3gemm_fused_inst.h"
#    include "moe_3gemm_gen_micro.hpp"
#    include "ocl_v2/utils/jitter.hpp"
#    include "primitive_inst.h"

namespace ov::intel_gpu::ocl {

namespace {

using namespace ov::intel_gpu::ocl;

// Bring shared onednn matmul wrappers into this anonymous namespace so existing call sites
// that reference `onednn_matmul`, `onednn_linear`, etc. unqualified continue to compile after
// the definitions moved to impls/onednn/grouped_matmul_helper.hpp.
using cldnn::onednn::onednn_linear;
using cldnn::onednn::onednn_matmul;

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
    case cldnn::data_types::f32:
        return dnnl::memory::data_type::f32;
    case cldnn::data_types::f16:
        return dnnl::memory::data_type::f16;
    case cldnn::data_types::i8:
        return dnnl::memory::data_type::s8;
    case cldnn::data_types::u8:
        return dnnl::memory::data_type::u8;
    case cldnn::data_types::i32:
        return dnnl::memory::data_type::s32;
    case cldnn::data_types::i4:
        return dnnl::memory::data_type::s4;
    case cldnn::data_types::u4:
        return dnnl::memory::data_type::u4;
    default:
        throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
    }
}

inline dnnl::algorithm moe_activation_to_dnnl_algo(ov::op::internal::MOE::Activation_type act) {
    switch (act) {
    case ov::op::internal::MOE::Activation_type::GEGLU_TANH:
        return dnnl::algorithm::eltwise_gelu_tanh;
    case ov::op::internal::MOE::Activation_type::GEGLU_ERF:
        return dnnl::algorithm::eltwise_gelu_erf;
    case ov::op::internal::MOE::Activation_type::SWIGLU:
    default:
        return dnnl::algorithm::eltwise_swish;
    }
}

class MoE3GemmSwigluGather : public KernelGenerator {
public:
    MoE3GemmSwigluGather() : KernelGenerator("moe_3gemm_swiglu_fuse", "gather") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();
        jit.make("GATHER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillMaskGen : public KernelGenerator {
public:
    MoE3GemmSwigluPrefillMaskGen() : KernelGenerator("moe_mask_gen", "prefill_mask_gen") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        jit.make("INPUT0_TYPE", "int");   // topk_id
        jit.make("OUTPUT_TYPE", "int");   // tokens_per_expert
        jit.make("OUTPUT1_TYPE", "int");  // experts_info_start_idx
        jit.make("OUTPUT2_TYPE", "int");  // experts_id
        jit.make("OUTPUT3_TYPE", "int");  // tokens_lens_per_expert
        jit.make("OUTPUT4_TYPE", "int");  // num_actual_used_experts

        auto& config = desc->_config;
        jit.make("NUM_EXPERTS_PER_TOKEN", config.top_k);
        jit.make("SET_TOKEN_LEN", 1);
        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

static size_t get_seq_len(cldnn::layout& layout) {
    auto shape = layout.get_shape();
    size_t seq_len = static_cast<size_t>(shape[0]);
    if (shape.size() >= 3) {
        seq_len = static_cast<size_t>(shape[0] * shape[1]);
    }
    return seq_len;
}

static size_t get_vec_size(const RuntimeParams& params) {
    const auto& input = params.get_input_layout(0);
    size_t vec_size = 1;
    switch (input.data_type) {
    case ov::element::i8:
    case ov::element::u8:
        vec_size = 16;
        break;
    case ov::element::f16:
        vec_size = 8;
        break;
    case ov::element::f32:
    case ov::element::i32:
        vec_size = 4;
        break;
    case ov::element::i64:
        vec_size = 2;
        break;
    default:
        vec_size = 1;
        break;
    }
    return vec_size;
}

static auto calc_thread_count(RuntimeParams& params, const size_t vector_size, const size_t hidden_size) {
    auto max_wgs = params.get_program().get_engine().get_device_info().max_work_group_size;
    const uint64_t threads_needed = (hidden_size + vector_size - 1) / vector_size;
    size_t local_threads_needed = std::min(threads_needed, max_wgs);
    size_t batches_per_thread = 1;
    size_t unaligned_elements = 0;

    if (threads_needed <= max_wgs) {
        batches_per_thread = 1;
        unaligned_elements = hidden_size % vector_size;
    } else {
        batches_per_thread = (threads_needed + max_wgs - 1) / max_wgs;
        auto new_block_size = batches_per_thread * vector_size;
        unaligned_elements = hidden_size % new_block_size;

        local_threads_needed = hidden_size / new_block_size;
        auto partialblock = (hidden_size % new_block_size != 0) ? 1 : 0;
        local_threads_needed += partialblock;
    }

    return std::tuple{local_threads_needed, batches_per_thread, unaligned_elements};
}
class MoE3GemmSwigluPrefillGather : public KernelGenerator {
public:
    explicit MoE3GemmSwigluPrefillGather(bool use_grouped_gemm = false)
        : KernelGenerator("moe_gather_ref", "prefill_gather"),
          m_use_grouped_gemm(use_grouped_gemm) {}

protected:
    bool m_use_grouped_gemm;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        // auto& engine = params.prog->get_engine();
        // const auto& info = engine.get_device_info();

        auto hidden_size = desc->_config.hidden_size;
        auto block_size = get_vec_size(params);
        auto [local_threads_count, batches_per_thread, unaligned_elements] = calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);

        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", block_size);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);
        jit.make("UNALIGNED_ELEMENTS", unaligned_elements);

        jit.make("INPUT0_TYPE", "half");
        jit.make("INPUT1_TYPE", "int");
        jit.make("OUTPUT_TYPE", "half");
        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");
        if (m_use_grouped_gemm)
            jit.make("ONEDNN_GROUPED_GEMM_USED", 1);

        GPU_DEBUG_TRACE_DETAIL << "MoE3GemmSwigluPrefillGather::get_jit_constants():  hidden_size: " << hidden_size << ", block_size: " << block_size
                               << ", local_threads_count: " << local_threads_count << ", batches_per_thread: " << batches_per_thread
                               << ", unaligned_elements: " << unaligned_elements << std::endl;

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillSwiglu : public KernelGenerator {
public:
    explicit MoE3GemmSwigluPrefillSwiglu(bool use_grouped_gemm = false)
        : KernelGenerator("moe_3gemm_swiglu_fuse", "prefill_swiglu"),
          m_use_grouped_gemm(use_grouped_gemm) {}

protected:
    bool m_use_grouped_gemm;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();

        jit.make("PREFILL_SWIGLU_ENABLE", 1);
        jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
        jit.make("INTERMEDIA_SIZE", desc->_config.inter_size);
        jit.make("MOE_DTYPE", "half");
        if (desc->_config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
            jit.make("GATE_ACT_GELU_TANH", 1);
        } else if (desc->_config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
            jit.make("GATE_ACT_GELU_ERF", 1);
        }
        if (m_use_grouped_gemm)
            jit.make("ONEDNN_GROUPED_GEMM_USED", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluPrefillScatterReduce : public KernelGenerator {
public:
    explicit MoE3GemmSwigluPrefillScatterReduce(bool use_grouped_gemm = false)
        : KernelGenerator("moe_scatter_reduction_opt", "moe_scatter_reduction_ref"),
          m_use_grouped_gemm(use_grouped_gemm) {}

protected:
    bool m_use_grouped_gemm;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        // auto& engine = params.prog->get_engine();
        // const auto& info = engine.get_device_info();

        auto hidden_size = desc->_config.hidden_size;
        auto block_size = 4;
        auto [local_threads_count, batches_per_thread, unaligned_elements] = calc_thread_count(const_cast<RuntimeParams&>(params), block_size, hidden_size);

        jit.make("OPTIONAL_SHAPE_INFO_ARG", "");
        jit.make("ACTIVE_EXPERTS", desc->_config.top_k);
        jit.make("HIDDEN_SIZE", hidden_size);
        jit.make("VEC_BLK_SIZE", 4);
        jit.make("BATCHES_PER_THREAD", batches_per_thread);
        jit.make("SET_ACTUAL_USED_EXPERTS_NUM", 1);

        jit.make("INPUT0_TYPE", "half");  // mlp_down output
        jit.make("INPUT1_TYPE", "int");   // expert indices per token
        jit.make("INPUT2_TYPE", "half");  // experts router weights
        jit.make("INPUT3_TYPE", "int");   // tokens per expert
        jit.make("INPUT4_TYPE", "int");   // expert start offsets
        jit.make("INPUT5_TYPE", "int");   // tokens len for experts
        jit.make("INPUT6_TYPE", "int");   // expert id
        jit.make("OUTPUT_TYPE", "half");  // output
        if (m_use_grouped_gemm)
            jit.make("ONEDNN_GROUPED_GEMM_USED", 1);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

class MoE3GemmSwigluScatter : public KernelGenerator {
public:
    MoE3GemmSwigluScatter() : KernelGenerator("moe_3gemm_swiglu_fuse", "index_add") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        jit.make("SCATTER_ENABLE", 1);
        jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
        jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
        jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

// Performance tuning parameters
#    define N_BLOCK      4
#    define SUBGROUP_NUM 8

static void add_common_consts(const RuntimeParams& params, JitConstants& jit) {
    auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
    auto& engine = params.prog->get_engine();
    const auto& info = engine.get_device_info();
    auto gate_up_group_size = desc->_config.group_size;
    auto down_group_size = desc->_config.group_size;
    if (desc->_config.group_size == std::numeric_limits<size_t>::max()) {
        gate_up_group_size = desc->_config.hidden_size;
        down_group_size = desc->_config.inter_size;
    }

    GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt: group_size=" << desc->_config.group_size << ", gate_up_group_size=" << gate_up_group_size
                           << ", down_group_size=" << down_group_size << std::endl;

    // Validate GEMV kernel compatibility: ELEMS_PER_LANE = FAKE_GROUP_SIZE / SUBGROUP_SIZE.
    // The u2 kernels have an ELEMS_PER_LANE==1 branch (group_size 32 on Xe2); all other
    // dtypes require >= 2 — smaller values have no kernel branch and would silently
    // produce wrong results.
    const ov::element::Type weight_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type;
    {
        const size_t sg = (info.arch >= gpu_arch::xe2) ? 32u : 16u;
        const size_t fake_gs = std::min(gate_up_group_size, size_t{128});
        const size_t min_fake_gs = (weight_dt == ov::element::u2) ? sg : 2 * sg;
        OPENVINO_ASSERT(fake_gs >= min_fake_gs,
                        "MoE GEMV kernel does not support group_size=",
                        gate_up_group_size,
                        " on this hardware (SUBGROUP_SIZE=",
                        sg,
                        "). Minimum supported group_size is ",
                        min_fake_gs,
                        ". Use a larger quantization group size.");
    }

    jit.make("MAX_TOPK", desc->_config.top_k);
    jit.make("EXPERT_NUM", desc->_config.num_expert);
    jit.make("HIDDEN_SIZE", desc->_config.hidden_size);
    jit.make("INTERMEDIATE_SIZE", desc->_config.inter_size);
    jit.make("N_BLOCK", N_BLOCK);
    jit.make("SUBGROUP_SIZE", info.arch >= gpu_arch::xe2 ? 32 : 16);
    jit.make("SUBGROUP_NUM", SUBGROUP_NUM);
    jit.make("GATE_UP_GROUP_SIZE", gate_up_group_size);
    jit.make("DOWN_GROUP_SIZE", down_group_size);
    jit.make("MOE_DTYPE", params.get_input_layout(0).data_type == ov::element::f16 ? "half" : "float");
    jit.make("MOE_DTYPE_SIZE", params.get_input_layout(0).data_type == ov::element::f16 ? 2 : 4);
    jit.make("HAS_ZP", desc->_config.has_zp ? 1 : 0);
    if (desc->_config.has_zp) {
        // MOE_ZP_SCALAR: at least one GEMM (u2 mixed-precision) carries a single broadcast
        // zp element. The u2 GEMV reads it directly; per-channel (u8) GEMMs ignore this and
        // index zp per (group, channel). Set it from whichever GEMM has a scalar zp.
        for (const auto zp_idx : {MOE3GemmInputIndex::ZP_0, MOE3GemmInputIndex::ZP_1, MOE3GemmInputIndex::ZP_2}) {
            const auto& zp_layout = params.get_input_layout(static_cast<size_t>(zp_idx));
            if (zp_layout.count() == 1) {
                OPENVINO_ASSERT(zp_layout.data_type == ov::element::i8 || zp_layout.data_type == ov::element::u8,
                                "Scalar MoE zp must be i8/u8, got ",
                                zp_layout.data_type);
                jit.make("MOE_ZP_SCALAR", 1);
                jit.make("MOE_ZP_SCALAR_DT", zp_layout.data_type == ov::element::i8 ? "char" : "uchar");
                break;
            }
        }
    }

    bool is_signed_weight = (weight_dt == ov::element::i4 || weight_dt == ov::element::i8);
    jit.make("WEIGHT_IS_SIGNED", is_signed_weight ? 1 : 0);
    // auto scale_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0)).data_type;
    // auto zp_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_0)).data_type;
    if (weight_dt == ov::element::u4 || weight_dt == ov::element::i4) {
        jit.make("WEIGHT_COMPRESSEION_DT", 0);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::u2) {
        // u2: 4 values per byte, LSB-first. Served by the batched GEMV kernels for all
        // token counts (micro-gemm and oneDNN have no u2 support).
        OPENVINO_ASSERT(desc->_config.hidden_size % 4 == 0 && desc->_config.inter_size % 4 == 0,
                        "MoE u2 weights require hidden_size and inter_size to be multiples of 4, got hidden_size=",
                        desc->_config.hidden_size,
                        ", inter_size=",
                        desc->_config.inter_size);
        jit.make("WEIGHT_COMPRESSEION_DT", 3);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::u8 || weight_dt == ov::element::i8) {
        jit.make("WEIGHT_COMPRESSEION_DT", 1);
        jit.make("MOE_WEI_DT", "uchar");
        jit.make("MOE_SCALE_DT", "half");
        jit.make("MOE_ZP_DT", "uchar");
    } else if (weight_dt == ov::element::f16) {
        jit.make("WEIGHT_COMPRESSEION_DT", 2);
        jit.make("MOE_WEI_DT", "half");
        jit.make("MOE_SCALE_DT", "half");  // not use
        jit.make("MOE_ZP_DT", "half");     // not use
    }

    // A3: per-GEMM weight dtype codes (gate/up/down may differ under NNCF mixed precision).
    // Code mirrors WEIGHT_COMPRESSEION_DT: 0=u4/i4, 1=u8/i8, 2=f16, 3=u2.
    const auto moe_weight_code = [](ov::element::Type dt) -> int {
        if (dt == ov::element::u4 || dt == ov::element::i4)
            return 0;
        if (dt == ov::element::f16)
            return 2;
        if (dt == ov::element::u2)
            return 3;
        return 1;  // u8/i8 (byte) path
    };
    const auto gate_wdt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type;
    const auto up_wdt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1)).data_type;
    const auto down_wdt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2)).data_type;
    jit.make("GATE_WEIGHT_DT", moe_weight_code(gate_wdt));
    jit.make("UP_WEIGHT_DT", moe_weight_code(up_wdt));
    jit.make("DOWN_WEIGHT_DT", moe_weight_code(down_wdt));
    // Per-GEMM scalar-zp flags: a single-element zp is broadcast in-kernel (u2 GEMMs),
    // vs per-channel/group zp indexed by (group, channel) (u8 GEMMs).
    if (desc->_config.has_zp) {
        jit.make("GATE_ZP_SCALAR", params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_0)).count() == 1 ? 1 : 0);
        jit.make("UP_ZP_SCALAR", params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_1)).count() == 1 ? 1 : 0);
        jit.make("DOWN_ZP_SCALAR", params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_2)).count() == 1 ? 1 : 0);
    } else {
        jit.make("GATE_ZP_SCALAR", 0);
        jit.make("UP_ZP_SCALAR", 0);
        jit.make("DOWN_ZP_SCALAR", 0);
    }
}

class MoE3GemmSwigluMLPGateUp : public KernelGenerator {
public:
    MoE3GemmSwigluMLPGateUp(bool disable_shared_experts = false)
        : KernelGenerator("moe_3gemm_swiglu_mlp", "gate_up"),
          _disable_shared_experts(disable_shared_experts) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        add_common_consts(params, jit);
        jit.make("GATE_UP_ENABLE", 1);
        if (desc->_config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
            jit.make("GATE_ACT_GELU_TANH", 1);
        } else if (desc->_config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
            jit.make("GATE_ACT_GELU_ERF", 1);
        }
        if (!_disable_shared_experts && desc->_config.num_shared_expert > 0 &&
            params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            jit.make("SHARED_EXPERT_ENABLE", 1);
        } else {
            jit.make("SHARED_EXPERT_ENABLE", 0);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }

private:
    bool _disable_shared_experts;
};

class MoE3GemmSwigluMLPDown : public KernelGenerator {
public:
    MoE3GemmSwigluMLPDown(bool disable_shared_experts = false)
        : KernelGenerator("moe_3gemm_swiglu_mlp", "down"),
          _disable_shared_experts(disable_shared_experts) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        add_common_consts(params, jit);
        jit.make("DOWN_ENABLE", 1);
        if (!_disable_shared_experts && desc->_config.num_shared_expert > 0 &&
            params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            jit.make("SHARED_EXPERT_ENABLE", 1);
        } else {
            jit.make("SHARED_EXPERT_ENABLE", 0);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }

private:
    bool _disable_shared_experts;
};

class MoE3GemmSwigluMLPReduce : public KernelGenerator {
public:
    MoE3GemmSwigluMLPReduce(bool disable_shared_experts = false)
        : KernelGenerator("moe_3gemm_swiglu_mlp", "reduce"),
          _disable_shared_experts(disable_shared_experts) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        add_common_consts(params, jit);
        jit.make("REDUCE_ENABLE", 1);
        if (!_disable_shared_experts && desc->_config.num_shared_expert > 0 &&
            params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            jit.make("SHARED_EXPERT_ENABLE", 1);
        } else {
            jit.make("SHARED_EXPERT_ENABLE", 0);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }

private:
    bool _disable_shared_experts;
};

// u2 -> u4 weight unpack kernel (prefill only): micro-gemm and oneDNN have no u2
// dtype, so u2 expert weights are unpacked into u4 scratch buffers and the regular
// u4 grouped GEMM prefill path consumes them.
class MoE3GemmSwigluMLPU2Unpack : public KernelGenerator {
public:
    MoE3GemmSwigluMLPU2Unpack() : KernelGenerator("moe_3gemm_swiglu_mlp", "u2_unpack") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        add_common_consts(params, jit);
        jit.make("U2_UNPACK_ENABLE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

// Native u2 grouped GEMM for prefill: consumes the u2 expert weights directly, so the u4
// unpack scratch (12.08 GB on Qwen3.6-35B-A3B) is not needed at all. One instantiation per
// GEMM because K and N differ between gate/up (K=hidden, N=inter) and down (K=inter, N=hidden),
// and the kernel sizes its SLM staging buffer from K at compile time.
//
// This exists because neither alternative works: oneDNN's JIT GEMM generator (gemmstone) has no
// 2-bit type and hard-codes at most 2 elements per byte, and the batched GEMV is one work-group
// per (token, expert) pair so it re-reads an expert's weights once per token.
class MoE3GemmSwigluU2Gemm : public KernelGenerator {
public:
    // gemm_idx: 0 = gate, 1 = up, 2 = down.
    explicit MoE3GemmSwigluU2Gemm(int gemm_idx)
        : KernelGenerator("moe_3gemm_swiglu_mlp", "u2_gemm_" + std::to_string(gemm_idx)),
          m_gemm_idx(gemm_idx) {}

    // Tokens staged in SLM per work-group. The whole point of the kernel: one weight byte read
    // (and one shift/mask dequant) feeds TILE_M FMAs, taking weight arithmetic intensity from
    // 2 to 2*TILE_M MACs/byte. Bounded by SLM: TILE_M * K * sizeof(half) must stay under 64 KB,
    // i.e. TILE_M <= 16 at K=2048.
    static constexpr int TILE_M = 8;

    // ---- DPAS variant (see the long comment on the kernel) ----
    // Sub-group size 16 is mandatory, not a tuning knob: intel_sub_group_f16_f16_matrix_mad_k16
    // compiles at the xe2+ default of 32 and silently returns wrong results there. It therefore
    // gets its own macro; the DPAS kernel never reads SUBGROUP_SIZE.
    static constexpr int DPAS_SG = 16;
    static constexpr int DPAS_MSUB = 4;                   // float8 accumulators per sub-group
    static constexpr int DPAS_N_SG = 16;                  // sub-groups per work-group
    static constexpr int DPAS_TILE_M = DPAS_MSUB * 8;     // 32 token rows per block
    static constexpr int DPAS_N_PER_WG = DPAS_N_SG * 16;  // 256 output channels per work-group

    // Shape constraints of the block2d activation load and the fixed-width weight gather. All
    // three GEMMs must qualify or none may: they share one block list, whose tile_m differs
    // between the two variants (8 vs 32).
    static bool dpas_stage_ok(size_t k, size_t n, size_t group_size) {
        if (group_size != 32 && group_size != 64 && group_size != 128) {
            return false;  // the per-quant-group weight gather is a fixed-width vload
        }
        // block2d pitch is K*2 bytes and must be >= 64 and a multiple of 16; a quant group must
        // also not straddle a 16-deep DPAS step.
        if (k % 32 != 0 || k % group_size != 0) {
            return false;
        }
        return n % DPAS_N_PER_WG == 0;
    }
    static bool dpas_supported(size_t hidden_size, size_t inter_size, size_t gate_up_gs, size_t down_gs) {
        return dpas_stage_ok(hidden_size, inter_size, gate_up_gs) && dpas_stage_ok(inter_size, hidden_size, down_gs);
    }

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        add_common_consts(params, jit);
        jit.make("U2_GEMM_ENABLE", 1);

        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        const auto& cfg = desc->_config;
        const bool is_down = (m_gemm_idx == 2);
        const size_t k = is_down ? cfg.inter_size : cfg.hidden_size;
        const size_t n = is_down ? cfg.hidden_size : cfg.inter_size;

        // Same per-channel resolution as add_common_consts(), and it has to be done for BOTH
        // stages here rather than just this one: dpas_supported() must return the same answer for
        // gate/up and down, since they share a block list.
        size_t gate_up_gs = cfg.group_size;
        size_t down_gs = cfg.group_size;
        if (cfg.group_size == std::numeric_limits<size_t>::max()) {
            // per-channel quantization degenerates to one group over all of K
            gate_up_gs = cfg.hidden_size;
            down_gs = cfg.inter_size;
        }
        const size_t group_size = is_down ? down_gs : gate_up_gs;

        jit.make("U2_GEMM_K", k);
        jit.make("U2_GEMM_N", n);
        jit.make("U2_GEMM_GROUP_SIZE", group_size);
        jit.make("TILE_M", TILE_M);

        // Per-GEMM zero-point form. add_common_consts()'s MOE_ZP_SCALAR is set from whichever of
        // the three GEMMs has a scalar zp, which is the wrong question for a kernel instantiated
        // once per GEMM: a mixed layer can pair a scalar (INT2_SYM) gate with a per-group
        // (INT2_ASYM) down. Decide from this GEMM's own zp layout.
        if (cfg.has_zp) {
            const auto zp_idx = m_gemm_idx == 0   ? MOE3GemmInputIndex::ZP_0
                                : m_gemm_idx == 1 ? MOE3GemmInputIndex::ZP_1
                                                  : MOE3GemmInputIndex::ZP_2;
            const auto& zp_layout = params.get_input_layout(static_cast<size_t>(zp_idx));
            const bool zp_scalar = zp_layout.count() == 1;
            jit.make("U2_GEMM_ZP_SCALAR", zp_scalar ? 1 : 0);
            if (!zp_scalar) {
                // The kernel unpacks the zp 4 per byte. A u8 zp behind u2 weights would be read at
                // a quarter of its true stride and produce plausible-looking garbage, so refuse it
                // here instead of at the end of an eval run.
                OPENVINO_ASSERT(zp_layout.data_type == ov::element::u2,
                                "Per-group zp for a u2 MoE GEMM must be u2 (4 per byte), got ",
                                zp_layout.data_type);
                const size_t per_expert = n * (k / group_size);
                OPENVINO_ASSERT(per_expert != 0 && zp_layout.count() % per_expert == 0,
                                "Per-group MoE zp element count ",
                                zp_layout.count(),
                                " is not a multiple of N*K/group_size = ",
                                per_expert);
            }
        }

        if (dpas_supported(cfg.hidden_size, cfg.inter_size, gate_up_gs, down_gs)) {
            jit.make("U2_DPAS_ENABLE", 1);
            jit.make("U2_DPAS_SG", DPAS_SG);
            jit.make("U2_MSUB", DPAS_MSUB);
            jit.make("U2_N_SG", DPAS_N_SG);
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        // execute_stage() builds the descriptor from the buffers it is handed.
        return Arguments{};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }

private:
    int m_gemm_idx;
};

dnnl::memory convert2dnnl(const memory::ptr& ptr, const std::vector<int64_t>& dim, dnnl::memory::format_tag tag, int64_t offset = 0) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("convert2dnnl"));
    return ptr->get_onednn_memory(dnnl::memory::desc(dnnl::memory::dims(dim), convert_data_type(ptr->get_layout().data_type), tag), offset);
}

// Returns the byte count for `element_count` elements of the given layout's data type.
// Handles sub-byte types (u4/i4) that pack two elements per byte.
static int64_t get_bytes_count(int64_t element_count, const cldnn::layout& layout) {
    return static_cast<int64_t>(ov::element::Type(layout.data_type).bitwidth()) * element_count / 8;
}

class moe_3gemm_swiglu_opt_impl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoE3GemmSwigluImpl)
    Stage::Ptr gather = make_stage<MoE3GemmSwigluGather>();
    Stage::Ptr scatter = make_stage<MoE3GemmSwigluScatter>();
    Stage::Ptr mlp_gate_up = make_stage<MoE3GemmSwigluMLPGateUp>();
    Stage::Ptr mlp_down = make_stage<MoE3GemmSwigluMLPDown>();
    Stage::Ptr mlp_reduce = make_stage<MoE3GemmSwigluMLPReduce>();

    Stage::Ptr prefill_gather = make_stage<MoE3GemmSwigluPrefillGather>();
    Stage::Ptr micro_gemm_gate = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_GATE);
    Stage::Ptr micro_gemm_up = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_UP);
    Stage::Ptr micro_gemm_down = make_stage<MoE3GemmMicroGenerator>(MoE3GemmMicroKernelType::MLP_DOWN);
    Stage::Ptr prefill_swiglu = make_stage<MoE3GemmSwigluPrefillSwiglu>();
    Stage::Ptr prefill_scatter_reduce = make_stage<MoE3GemmSwigluPrefillScatterReduce>();
    Stage::Ptr prefill_mask_gen = make_stage<MoE3GemmSwigluPrefillMaskGen>();

    // Grouped GEMM path: same OCL kernels but compiled with ONEDNN_GROUPED_GEMM_USED=1
    Stage::Ptr grouped_gemm_prefill_gather = make_stage<MoE3GemmSwigluPrefillGather>(/*use_grouped_gemm=*/true);
    Stage::Ptr grouped_gemm_prefill_swiglu = make_stage<MoE3GemmSwigluPrefillSwiglu>(/*use_grouped_gemm=*/true);
    Stage::Ptr grouped_gemm_prefill_scatter_reduce = make_stage<MoE3GemmSwigluPrefillScatterReduce>(/*use_grouped_gemm=*/true);

    // u2 prefill: unpack u2 weights into u4 scratch for the grouped GEMM path
    Stage::Ptr u2_unpack = make_stage<MoE3GemmSwigluMLPU2Unpack>();

    // u2 prefill, native path: consumes u2 directly, no unpack and no u4 scratch.
    Stage::Ptr u2_gemm_gate = make_stage<MoE3GemmSwigluU2Gemm>(0);
    Stage::Ptr u2_gemm_up = make_stage<MoE3GemmSwigluU2Gemm>(1);
    Stage::Ptr u2_gemm_down = make_stage<MoE3GemmSwigluU2Gemm>(2);

    struct dnnl_weights {
        dnnl::memory weight;
        dnnl::memory scale;
        dnnl::memory zp;
        int ic, oc, ic_group_size;
    };

    // expert_mask result in cpu side
    struct expert_mask_cpu {
        std::vector<int8_t> pred_flag;
        // shape: [expert_num, batch_no]
        std::vector<std::vector<int>> batch;
        // shape: [expert_num, topk_no]
        std::vector<std::vector<int>> topk;
    };

    // store expert_mask for gpu kernel
    struct expert_mask_gpu {
        memory::ptr batch;
        memory::ptr topk;
    };

    struct moe_fusion_weights_base_addr {
        memory::ptr weight[3];  // gate/up/down weights, experts fusion
        memory::ptr scale[3];
        memory::ptr zp[3];
        memory::ptr bias[3];

        // Shared expert: Gate, Up, Down, ScalarGate
        memory::ptr shared_weight[4];
        memory::ptr shared_scale[4];
        memory::ptr shared_zp[4];
    } moe_fusion_weights;

    struct scratch_buffers {
        // softmax+topk
        memory::ptr topk_id;
        memory::ptr topk_weights;

        // fast single batch: scratch.up = up(x) * silu(gate(x))
        //                    scratch.y = down(scratch.up) * routing_weights
        memory::ptr up;
        memory::ptr y;
        // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        //         scratch.up = up(scratch.x)
        //         scratch.gate = gate(scratch.x) * scratch.up
        //         scratch.y = down(scratch.gate) * routing_weights
        memory::ptr x;
        memory::ptr routing_weights;
        memory::ptr gate;
        // buffers for batch and topk from cpu, each expert has one
        std::vector<expert_mask_gpu> expert_masks;

        moe_fusion_weights_base_addr moe_fusion_wei_addr;
        memory::ptr input_routing_weights;
        memory::ptr input_router_topk_idx;
        memory::ptr _expert_index_buffer;
    };

    std::vector<std::vector<dnnl_weights>> _dnnl_weights;
    int _hidden_size;
    int _intermediate_size;
    int _shared_intermediate_size;
    int _gate_up_group_size;
    int _down_group_size;
    std::shared_ptr<IExpertWeightProvider> _weight_provider;

    // --- OTD helper methods (used when _weight_provider->is_offloaded()) ---

    void set_otd_weight_pointers(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, scratch_buffers& scratch) {
        const auto& w = instance._weights;
        scratch.moe_fusion_wei_addr.weight[0] = w.gate_w;
        scratch.moe_fusion_wei_addr.scale[0] = w.gate_s;
        scratch.moe_fusion_wei_addr.zp[0] = w.gate_z;
        scratch.moe_fusion_wei_addr.weight[1] = w.up_w;
        scratch.moe_fusion_wei_addr.scale[1] = w.up_s;
        scratch.moe_fusion_wei_addr.zp[1] = w.up_z;
        scratch.moe_fusion_wei_addr.weight[2] = w.down_w;
        scratch.moe_fusion_wei_addr.scale[2] = w.down_s;
        scratch.moe_fusion_wei_addr.zp[2] = w.down_z;
    }

    size_t resident_slot_count() const {
        return _weight_provider->resident_capacity();
    }

    // --- Inline composition methods (replace virtual hooks) ---

    void bind_weights_on_first_exec(typed_primitive_inst<moe_3gemm_fused_compressed>& instance) {
        if (!_weight_provider->is_offloaded())
            return;
        if (!_weight_provider->is_bound()) {
            instance._weights.gate_w = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0));
            instance._weights.gate_z = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_0));
            instance._weights.gate_s = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0));

            instance._weights.up_w = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1));
            instance._weights.up_z = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_1));
            instance._weights.up_s = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_1));

            instance._weights.down_w = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2));
            instance._weights.down_z = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_2));
            instance._weights.down_s = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_2));
            _weight_provider->bind(instance._weights);
        }
    }

    void prepare_weight_pointers(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, scratch_buffers& scratch) {
        if (_weight_provider->is_offloaded()) {
            set_otd_weight_pointers(instance, scratch);
        } else {
            scratch.moe_fusion_wei_addr.weight[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0));
            scratch.moe_fusion_wei_addr.scale[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0));
            scratch.moe_fusion_wei_addr.zp[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_0));
            scratch.moe_fusion_wei_addr.weight[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1));
            scratch.moe_fusion_wei_addr.scale[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_1));
            scratch.moe_fusion_wei_addr.zp[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_1));
            scratch.moe_fusion_wei_addr.weight[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2));
            scratch.moe_fusion_wei_addr.scale[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SCALE_2));
            scratch.moe_fusion_wei_addr.zp[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::ZP_2));
        }

        // Shared expert weight pointers (indices 12-21)
        if (instance.dependencies().size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            scratch.moe_fusion_wei_addr.shared_weight[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT));
            scratch.moe_fusion_wei_addr.shared_scale[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_SCALE));
            scratch.moe_fusion_wei_addr.shared_zp[0] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_ZP));
            scratch.moe_fusion_wei_addr.shared_weight[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_UP_WEIGHT));
            scratch.moe_fusion_wei_addr.shared_scale[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_UP_SCALE));
            scratch.moe_fusion_wei_addr.shared_zp[1] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_UP_ZP));
            scratch.moe_fusion_wei_addr.shared_weight[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_DOWN_WEIGHT));
            scratch.moe_fusion_wei_addr.shared_scale[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_DOWN_SCALE));
            scratch.moe_fusion_wei_addr.shared_zp[2] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_DOWN_ZP));
            scratch.moe_fusion_wei_addr.shared_weight[3] = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_GATE_WEIGHT));
        }
    }

    void on_before_batched_gemv(cldnn::stream& stream,
                                typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                scratch_buffers& scratch,
                                size_t topk_count,
                                bool& needs_fallback) {
        if (!_weight_provider->is_offloaded()) {
            needs_fallback = false;
            return;
        }
        stream.finish();

        std::vector<uint32_t> expert_ids(topk_count);
        scratch.topk_id->copy_to(stream, expert_ids.data(), 0, 0, topk_count * sizeof(uint32_t), true);

        auto lease = _weight_provider->try_acquire_simultaneous(expert_ids, stream);
        if (!lease) {
            if (auto* perf = moe_otd::get_perf_counters())
                perf->batched_fallbacks.fetch_add(1, std::memory_order_relaxed);
            needs_fallback = true;
            return;
        }

        // Write remapped slot IDs to GPU buffer
        auto& engine = instance.get_network().get_engine();
        const size_t topk_bytes = topk_count * sizeof(uint32_t);
        if (!scratch._expert_index_buffer || scratch._expert_index_buffer->size() < topk_bytes) {
            auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(topk_bytes)}, ov::element::i8, cldnn::format::bfyx);
            scratch._expert_index_buffer = engine.allocate_memory(layout, allocation_type::usm_host, false);
        }
        std::vector<uint32_t> slots_u32(lease->size());
        for (size_t i = 0; i < lease->size(); i++)
            slots_u32[i] = static_cast<uint32_t>((*lease)[i]);
        scratch._expert_index_buffer->copy_from(stream, slots_u32.data(), 0, 0, topk_bytes, true);

        set_otd_weight_pointers(instance, scratch);
        needs_fallback = false;
    }

    void on_before_prefill(cldnn::stream& stream,
                           typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                           scratch_buffers& scratch,
                           memory::ptr& batch_mem_ptr,
                           size_t topk_count,
                           bool& needs_fallback) {
        if (!_weight_provider->is_offloaded()) {
            needs_fallback = false;
            return;
        }

        std::vector<uint32_t> expert_ids(topk_count);
        batch_mem_ptr->copy_to(stream, expert_ids.data(), 0, 0, topk_count * sizeof(uint32_t), true);

        auto lease = _weight_provider->try_acquire_simultaneous(expert_ids, stream);
        if (!lease) {
            if (auto* perf = moe_otd::get_perf_counters())
                perf->grouped_fallbacks.fetch_add(1, std::memory_order_relaxed);
            needs_fallback = true;
            return;
        }

        // Write remapped slot IDs to GPU buffer
        auto& engine = instance.get_network().get_engine();
        const size_t topk_bytes = topk_count * sizeof(uint32_t);
        if (!scratch._expert_index_buffer || scratch._expert_index_buffer->size() < topk_bytes) {
            auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(topk_bytes)}, ov::element::i8, cldnn::format::bfyx);
            scratch._expert_index_buffer = engine.allocate_memory(layout, allocation_type::usm_host, false);
        }
        std::vector<uint32_t> slots_u32(lease->size());
        for (size_t i = 0; i < lease->size(); i++)
            slots_u32[i] = static_cast<uint32_t>((*lease)[i]);
        scratch._expert_index_buffer->copy_from(stream, slots_u32.data(), 0, 0, topk_bytes, true);
        batch_mem_ptr = scratch._expert_index_buffer;
        needs_fallback = false;
    }

    bool on_load_expert_weights(size_t expert_no, typed_primitive_inst<moe_3gemm_fused_compressed>& instance, dnnl::stream& dnn_stream) {
        if (!_weight_provider->is_offloaded())
            return false;

        dnn_stream.wait();

        auto& stream = instance.get_network().get_stream();
        auto& dnnl_weights = _dnnl_weights[expert_no];
        auto lru_expert_no = static_cast<int64_t>(_weight_provider->acquire_one(static_cast<uint32_t>(expert_no), stream));
        auto& params = instance._weights;

#    define CONVERT_DNNL_OTD(name, i)                                                                                                                  \
        int64_t wei_offset##i = lru_expert_no * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc, params.name##_w->get_layout());               \
        int64_t scale_offset##i =                                                                                                                      \
            lru_expert_no * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size, params.name##_s->get_layout());   \
        int64_t zp_offset##i =                                                                                                                         \
            lru_expert_no * get_bytes_count(dnnl_weights[i].ic * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size, params.name##_z->get_layout());   \
        dnnl_weights[i].weight = convert2dnnl(params.name##_w, {dnnl_weights[i].ic, dnnl_weights[i].oc}, dnnl::memory::format_tag::ba, wei_offset##i); \
        dnnl_weights[i].scale = convert2dnnl(params.name##_s,                                                                                          \
                                             {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},                                 \
                                             dnnl::memory::format_tag::ab,                                                                             \
                                             scale_offset##i);                                                                                         \
        dnnl_weights[i].zp = convert2dnnl(params.name##_z,                                                                                             \
                                          {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},                                    \
                                          dnnl::memory::format_tag::ab,                                                                                \
                                          zp_offset##i);
        CONVERT_DNNL_OTD(gate, 0)
        CONVERT_DNNL_OTD(up, 1)
        CONVERT_DNNL_OTD(down, 2)
#    undef CONVERT_DNNL_OTD
        return true;
    }

    int get_num_grouped_experts(int num_total_experts) {
        return _weight_provider->is_offloaded() ? static_cast<int>(resident_slot_count()) : num_total_experts;
    }

    bool build_grouped_mask_otd(cldnn::stream& stream,
                                typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                scratch_buffers& scratch,
                                memory::ptr& batch_mem_ptr,
                                size_t token_num,
                                int max_topk,
                                int num_grouped_experts,
                                std::vector<int32_t>& tokens_per_expert_cpu,
                                std::vector<int32_t>& tokens_lens_per_expert_cpu,
                                std::vector<int32_t>& experts_id_cpu,
                                std::vector<int32_t>& grouped_offsets_cpu,
                                int& num_actually_used_experts,
                                const std::vector<cldnn::event::ptr>& events) {
        if (!_weight_provider->is_offloaded())
            return false;

        stream.finish();  // ensure routing kernel has written topk_ids

        size_t topk_count = static_cast<size_t>(token_num) * max_topk;
        std::vector<uint32_t> raw_topk_ids(topk_count);
        batch_mem_ptr->copy_to(stream, raw_topk_ids.data(), 0, 0, topk_count * sizeof(uint32_t), true);

        // Use lease API for capacity-safe simultaneous acquisition.
        // If unique experts exceed resident capacity, returns nullopt → caller falls back.
        auto lease = _weight_provider->try_acquire_simultaneous(raw_topk_ids, stream);
        if (!lease) {
            GPU_DEBUG_TRACE_DETAIL << "exec_prefill_grouped_gemm OTD: unique experts exceed resident_slots=" << resident_slot_count()
                                   << ", falling back to per-expert onednn loop" << std::endl;
            if (auto* perf = moe_otd::get_perf_counters())
                perf->grouped_fallbacks.fetch_add(1, std::memory_order_relaxed);
            return false;  // Caller checks is_offloaded() to distinguish from non-OTD
        }

        // Remap using lease slots
        std::vector<uint32_t> remapped(topk_count);
        for (size_t i = 0; i < topk_count; i++)
            remapped[i] = static_cast<uint32_t>((*lease)[i]);

        // Build per-slot token lists sorted by LRU slot index
        std::vector<std::vector<int32_t>> slot_tokens(num_grouped_experts);
        for (size_t i = 0; i < topk_count; i++) {
            int32_t slot = static_cast<int32_t>(remapped[i]);
            int32_t token_idx = static_cast<int32_t>(i / max_topk);
            slot_tokens[slot].push_back(token_idx);
        }

        // Build grouped_offsets and tokens_per_expert sorted by slot
        int tokens_iter = 0;
        int experts_iter = 0;
        int32_t running_offset = 0;
        for (int s = 0; s < num_grouped_experts; s++) {
            auto n = static_cast<int32_t>(slot_tokens[s].size());
            running_offset += n;
            grouped_offsets_cpu[s] = running_offset;
            if (n > 0) {
                experts_id_cpu[experts_iter] = s;
                tokens_lens_per_expert_cpu[experts_iter] = n;
                ++experts_iter;
                ++num_actually_used_experts;
                for (auto t : slot_tokens[s])
                    tokens_per_expert_cpu[tokens_iter++] = t;
            }
        }

        // Upload remapped topk_ids to GPU for scatter_reduce kernel
        auto& engine = instance.get_network().get_engine();
        size_t topk_bytes = topk_count * sizeof(uint32_t);
        if (!scratch._expert_index_buffer || scratch._expert_index_buffer->size() < topk_bytes) {
            auto remap_layout = batch_mem_ptr->get_layout();
            scratch._expert_index_buffer = engine.allocate_memory(remap_layout, allocation_type::usm_host, false);
        }
        scratch._expert_index_buffer->copy_from(stream, remapped.data(), 0, 0, topk_bytes, true);
        batch_mem_ptr = scratch._expert_index_buffer;

        set_otd_weight_pointers(instance, scratch);
        return true;
    }

    void on_after_grouped_gemm(cldnn::stream& stream) {
        if (_weight_provider->is_offloaded())
            stream.get_onednn_stream().wait();
    }

    void on_after_exec_sync(cldnn::stream& stream) {
        if (_weight_provider->is_offloaded())
            stream.finish();
    }

    bool should_pre_zero_output() {
        // OTD may fallback to exec_prefill_onednn which accumulates via index_add
        return !use_grouped_gemm_prefill || _weight_provider->is_offloaded();
    }

    bool should_validate_dnnl_weights() {
        // OTD doesn't validate at init time because weights aren't bound yet
        return !_weight_provider->is_offloaded();
    }

    void on_before_grouped_gather(cldnn::stream& stream) {
        if (_weight_provider->is_offloaded())
            stream.finish();
    }
    ov::op::internal::MOE::Activation_type _activation_type = ov::op::internal::MOE::Activation_type::SWIGLU;

    bool _has_shared_expert = false;
    // Shared expert primitives
    std::shared_ptr<onednn_linear> _shared_gate_proj;
    std::shared_ptr<onednn_linear> _shared_up_proj;
    std::shared_ptr<onednn_linear> _shared_down_proj;
    std::shared_ptr<onednn_linear> _shared_gate_gate_proj;  // The scalar gate for shared expert

    // Instance-specific flags (not static to avoid race conditions)
    bool use_micro_gemm_prefill = false;
    bool use_gpu_mask_gen_prefill = false;
    bool use_grouped_gemm_prefill = false;
    bool _weights_u2 = false;                     // any u2 GEMM: batched GEMV for decode, u4-unpack + grouped GEMM for prefill
    bool _gemm_weights_u2[3] = {false, false, false};  // per-GEMM (gate/up/down) u2 flags
    bool _shared_weights_u2[4] = {false, false, false, false};  // per-projection (gate/up/down/scalar-gate) shared-expert u2 flags
    // u2 prefill scratch: unpacked u4 copies of this layer's expert weights (and zp).
    // Allocated lazily on first prefill and reused across calls (unpack_u2_weights_for_prefill).
    memory::ptr _u2_unpack_weight[3];
    memory::ptr _u2_unpack_zp[3];
    // Same for the shared expert's weights/zp; consumed by the oneDNN shared-expert
    // primitives built in init_shared_primitives.
    memory::ptr _u2_unpack_shared_weight[4];
    memory::ptr _u2_unpack_shared_zp[4];

    // Identifies the source buffer a u4 unpack scratch was last filled from, so the unpack
    // kernel runs once instead of on every prefill. The unpack is a pure function of a
    // *constant* weight/zp buffer: OTD (expert offload) is rejected for u2 in the ctor, so
    // nothing rebinds these mid-run. Keyed on buffer identity + byte size so a rebind still
    // re-runs the unpack; a freshly (re)allocated destination also forces a re-run.
    struct unpack_src_key {
        const void* src = nullptr;
        size_t bytes = 0;

        bool matches(const memory::ptr& src_mem) const {
            return src != nullptr && src == static_cast<const void*>(src_mem.get()) && bytes == src_mem->size();
        }
        void set(const memory::ptr& src_mem) {
            src = static_cast<const void*>(src_mem.get());
            bytes = src_mem->size();
        }
        void reset() {
            src = nullptr;
            bytes = 0;
        }
    };
    // Work-block list for the native u2 GEMM: {expert_id, token_start, n_tokens} per block.
    // Rebuilt every prefill because the routing (and therefore the per-expert token counts)
    // changes with the input; grown in place, never shrunk.
    memory::ptr _u2_gemm_blocks;

    unpack_src_key _u2_unpack_weight_key[3];
    unpack_src_key _u2_unpack_zp_key[3];
    unpack_src_key _u2_unpack_shared_weight_key[4];
    unpack_src_key _u2_unpack_shared_zp_key[4];

    // The native u2 GEMM may only serve a layer whose three expert GEMMs are ALL u2. `_weights_u2`
    // is the OR of the three (ctor), so using it here would feed a u8/u4 weight buffer to a kernel
    // that reads a u2 expert stride (N*K/4) and unpacks every byte into four 2-bit values — silent
    // garbage on any mixed-precision layer, which is exactly what the per-GEMM-dtype (A3) work
    // exists to support. The unpack decision in execute() and the routing decision in
    // exec_prefill_grouped_gemm() must use this same predicate: if they disagree, a mixed layer
    // ends up on the oneDNN u4 path with routed weights that were never unpacked.
    bool use_native_u2_prefill() const {
        return _gemm_weights_u2[0] && _gemm_weights_u2[1] && _gemm_weights_u2[2];
    }

    // Which variant of moe_u2_gemm was JIT-compiled. This mirrors the decision the kernel
    // generator makes from the same four numbers, and the two MUST agree: the variants differ in
    // sub-group size, work-group shape and block tile_m, so a disagreement dispatches the wrong
    // geometry rather than failing to build.
    bool use_u2_dpas() const {
        return MoE3GemmSwigluU2Gemm::dpas_supported(static_cast<size_t>(_hidden_size),
                                                    static_cast<size_t>(_intermediate_size),
                                                    static_cast<size_t>(_gate_up_group_size),
                                                    static_cast<size_t>(_down_group_size));
    }

    bool has_u2_shared_weights() const {
        return _shared_weights_u2[0] || _shared_weights_u2[1] || _shared_weights_u2[2] || _shared_weights_u2[3];
    }
    size_t batched_gemv_threshold = 32;  // token_num <= threshold uses batched GEMV path

    moe_3gemm_swiglu_opt_impl() : PrimitiveImplOCL(moe_3gemm_swiglu_opt::get_type_info_static()) {}
    moe_3gemm_swiglu_opt_impl(const program_node& node, const RuntimeParams& params) : moe_3gemm_swiglu_opt_impl() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoE3GemmRuntimeParams>();
        }
        init(node.as<moe_3gemm_fused_compressed>().get_primitive());

        const auto& config = params.get_program().get_config();

        // micro_gemm is better than gemm, default to use it
        use_micro_gemm_prefill = config.get_moe_use_micro_gemm_prefill();
        // gpu mask gen kernel performance is worse than cpu mask gen, default is off
        use_gpu_mask_gen_prefill = config.get_moe_use_gpu_mask_gen_prefill();

        auto& engine = params.prog->get_engine();
        const auto& info = engine.get_device_info();
        if (info.arch < gpu_arch::xe2) {
            use_micro_gemm_prefill = false;
        }
        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt_impl(): use_micro_gemm_prefill=" << use_micro_gemm_prefill
                               << ", arch=" << static_cast<int>(info.arch) << std::endl;

        // Remove this limitation once micro_gemm kernels has supported i8/u8 weights.
        const auto& weight_dt = params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type;
        if (weight_dt != data_types::u4 && weight_dt != data_types::i4 && use_micro_gemm_prefill) {
            use_micro_gemm_prefill = false;
        }

        // u2 (2-bit) weights: micro-gemm (gemmstone) and oneDNN have no u2 dtype, so the
        // u4-only prefill paths cannot consume u2 data directly. Decode (small token
        // counts) is served by the hand-written batched GEMV kernels; prefill unpacks
        // the u2 weights into u4 scratch buffers and runs the grouped GEMM path.
        // In NNCF mixed-precision layers ANY u2 GEMM (gate/up/down) selects this split (A3).
        _gemm_weights_u2[0] = (weight_dt == data_types::u2);
        _gemm_weights_u2[1] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1)).data_type == data_types::u2);
        _gemm_weights_u2[2] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2)).data_type == data_types::u2);
        _weights_u2 = _gemm_weights_u2[0] || _gemm_weights_u2[1] || _gemm_weights_u2[2];

        // Shared expert per-projection u2 flags (gate/up/down/scalar-gate). Under NNCF
        // mixed precision any subset of them may be u2; u2 shared weights are unpacked
        // into u4 scratch for the oneDNN shared-expert primitives (init_shared_primitives).
        if (params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            _shared_weights_u2[0] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)).data_type == data_types::u2);
            _shared_weights_u2[1] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_UP_WEIGHT)).data_type == data_types::u2);
            _shared_weights_u2[2] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_DOWN_WEIGHT)).data_type == data_types::u2);
            _shared_weights_u2[3] = (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_GATE_WEIGHT)).data_type == data_types::u2);
        }

        // grouped_gemm path: single OneDNN grouped matmul per GEMM layer (all experts at once).
        // micro_gemm takes priority; grouped_gemm falls back to onednn loop by default.
        use_grouped_gemm_prefill = config.get_moe_use_grouped_gemm_prefill();
        if (_weights_u2) {
            // gemmstone has no u2 support at all; the grouped GEMM prefill consumes the
            // u2 weights via the u4-unpacked scratch (see unpack_u2_weights_for_prefill).
            use_micro_gemm_prefill = false;
            use_grouped_gemm_prefill = true;
        }
        // grouped_gemm supersedes micro_gemm
        if (use_grouped_gemm_prefill) {
            use_micro_gemm_prefill = false;
        }

        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt_impl(): use_grouped_gemm_prefill=" << use_grouped_gemm_prefill << std::endl;

        batched_gemv_threshold = config.get_moe_batched_gemv_threshold();
        if (batched_gemv_threshold == 0) {
            batched_gemv_threshold = 1;
        }
        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt_impl(): batched_gemv_threshold=" << batched_gemv_threshold << std::endl;

        // Weight provider: select based on OTD configuration.
        auto cur_moe = node.as<moe_3gemm_fused_compressed>().get_primitive();
        OPENVINO_ASSERT(!_weights_u2 || cur_moe->_otd.lru_expert_num == 0,
                        "moe_3gemm: u2 weights are not supported with OTD (expert offload) yet");
        if (cur_moe->_otd.lru_expert_num > 0) {
            _weight_provider = std::make_shared<OffloadExpertWeightProvider>(cur_moe->_otd.lru_expert_num,
                                                                             cur_moe->_config,
                                                                             cur_moe->_otd.weight_bin_offsets,
                                                                             cur_moe->_otd.weights_path);
        } else {
            _weight_provider = std::make_shared<ResidentExpertWeightProvider>();
        }

        // Don't change the order of stages
        add_stage(gather, params);
        add_stage(scatter, params);
        add_stage(mlp_gate_up, params);
        add_stage(mlp_down, params);
        add_stage(mlp_reduce, params);
        if (use_micro_gemm_prefill) {
            add_stage(prefill_mask_gen, params);
            add_stage(prefill_gather, params);
            add_stage(micro_gemm_gate, params);
            add_stage(micro_gemm_up, params);
            add_stage(prefill_swiglu, params);
            add_stage(micro_gemm_down, params);
            add_stage(prefill_scatter_reduce, params);
        }
        if (use_grouped_gemm_prefill) {
            add_stage(grouped_gemm_prefill_gather, params);
            add_stage(grouped_gemm_prefill_swiglu, params);
            add_stage(grouped_gemm_prefill_scatter_reduce, params);
        }
        if (_weights_u2 || has_u2_shared_weights()) {
            add_stage(u2_unpack, params);
            // Native u2 GEMM stages. Registered alongside the unpack so the impl can fall back
            // to the unpack + u4 grouped GEMM path if the native kernel fails at runtime.
            if (_weights_u2) {
                add_stage(u2_gemm_gate, params);
                add_stage(u2_gemm_up, params);
                add_stage(u2_gemm_down, params);
            }
        }
    }

    void init(const std::shared_ptr<const moe_3gemm_fused_compressed>& cur_moe) {
        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);
        _gate_up_group_size = static_cast<int>(cur_moe->_config.group_size);
        _down_group_size = static_cast<int>(cur_moe->_config.group_size);
        _activation_type = cur_moe->_config.activation_type;

        if (cur_moe->_config.group_size == std::numeric_limits<size_t>::max()) {
            _gate_up_group_size = static_cast<int>(cur_moe->_config.hidden_size);
            _down_group_size = static_cast<int>(cur_moe->_config.inter_size);
        }
        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] moe_3gemm_swiglu_opt prefill: group_size=" << cur_moe->_config.group_size
                               << ", gate_up_group_size=" << _gate_up_group_size << ", down_group_size=" << _down_group_size << std::endl;
    }

    void init_dnnl_weights(const std::shared_ptr<const moe_3gemm_fused_compressed>& cur_moe,
                           cldnn::engine& engine,
                           const struct moe_fusion_weights_base_addr& moe_fusion_wei_addr) {
        if (_dnnl_weights.size() == cur_moe->_config.num_expert)
            return;
        init(cur_moe);

        _dnnl_weights.resize(cur_moe->_config.num_expert);
        // Per-GEMM ic_group_size from scale shape; config.group_size can't represent gate/up vs down differing.
        const auto ic_group_size_from_scale = [](size_t ic, const cldnn::memory::ptr& scale_mem) {
            const auto& scale_shape = scale_mem->get_layout().get_shape();
            const size_t num_groups = (scale_shape.size() >= 3) ? scale_shape[2] : 1;
            return (num_groups <= 1) ? static_cast<int>(ic) : static_cast<int>(ic / num_groups);
        };
        for (size_t j = 0; j < cur_moe->_config.num_expert; j++) {
            auto& dnnl_weights = _dnnl_weights[j];
            dnnl_weights.resize(3);
            dnnl_weights[0].ic = _hidden_size;
            dnnl_weights[0].ic_group_size =
                moe_fusion_wei_addr.scale[0] ? ic_group_size_from_scale(_hidden_size, moe_fusion_wei_addr.scale[0]) : _gate_up_group_size;
            dnnl_weights[0].oc = _intermediate_size;
            dnnl_weights[1].ic = _hidden_size;
            dnnl_weights[1].ic_group_size =
                moe_fusion_wei_addr.scale[1] ? ic_group_size_from_scale(_hidden_size, moe_fusion_wei_addr.scale[1]) : _gate_up_group_size;
            dnnl_weights[1].oc = _intermediate_size;
            dnnl_weights[2].ic = _intermediate_size;
            dnnl_weights[2].ic_group_size =
                moe_fusion_wei_addr.scale[2] ? ic_group_size_from_scale(_intermediate_size, moe_fusion_wei_addr.scale[2]) : _down_group_size;
            dnnl_weights[2].oc = _hidden_size;
            if (should_validate_dnnl_weights()) {
                for (int i = 0; i < 3; i++) {
                    // Cross-check ic/ic_group_size against scale shape (drift caused u8 inf bug).
                    {
                        const auto& sshape = moe_fusion_wei_addr.scale[i]->get_layout().get_shape();
                        const size_t scale_num_groups = (sshape.size() >= 3) ? sshape[2] : 1;
                        OPENVINO_ASSERT(dnnl_weights[i].ic_group_size > 0, "moe_3gemm GEMM ", i, " ic_group_size must be > 0");
                        OPENVINO_ASSERT(dnnl_weights[i].ic % dnnl_weights[i].ic_group_size == 0,
                                        "moe_3gemm GEMM ",
                                        i,
                                        " ic=",
                                        dnnl_weights[i].ic,
                                        " not divisible by ic_group_size=",
                                        dnnl_weights[i].ic_group_size);
                        const auto expected_groups = dnnl_weights[i].ic / dnnl_weights[i].ic_group_size;
                        OPENVINO_ASSERT(static_cast<size_t>(expected_groups) == scale_num_groups,
                                        "moe_3gemm GEMM ",
                                        i,
                                        " ic_group_size=",
                                        dnnl_weights[i].ic_group_size,
                                        " (=> ",
                                        expected_groups,
                                        " groups) disagrees with scale num_groups=",
                                        scale_num_groups,
                                        " (scale shape=",
                                        sshape,
                                        ")");
                        if (cur_moe->_config.has_zp && moe_fusion_wei_addr.zp[i]) {
                            const auto& zshape = moe_fusion_wei_addr.zp[i]->get_layout().get_shape();
                            OPENVINO_ASSERT(zshape == sshape, "moe_3gemm GEMM ", i, " scale shape ", sshape, " does not match zp shape ", zshape);
                        }
                    }
                    // weight shape: [ic, oc], type: u4/i8
                    int64_t wei_offset =
                        j * get_bytes_count(static_cast<int64_t>(dnnl_weights[i].ic) * dnnl_weights[i].oc, moe_fusion_wei_addr.weight[i]->get_layout());
                    dnnl_weights[i].weight =
                        convert2dnnl(moe_fusion_wei_addr.weight[i], {dnnl_weights[i].ic, dnnl_weights[i].oc}, dnnl::memory::format_tag::ba, wei_offset);

                    // scale shape: [ic / ic_group_size, oc], type: f16
                    int64_t scale_offset = j * get_bytes_count(static_cast<int64_t>(dnnl_weights[i].ic) * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size,
                                                               moe_fusion_wei_addr.scale[i]->get_layout());
                    dnnl_weights[i].scale = convert2dnnl(moe_fusion_wei_addr.scale[i],
                                                         {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                         dnnl::memory::format_tag::ab,
                                                         scale_offset);

                    // zp shape: [ic / ic_group_size, oc], type: u4/i8
                    // Skip ZP memory allocation for symmetric quantization (has_zp=false) to save memory
                    if (cur_moe->_config.has_zp) {
                        int64_t zp_offset = j * get_bytes_count(static_cast<int64_t>(dnnl_weights[i].ic) * dnnl_weights[i].oc / dnnl_weights[i].ic_group_size,
                                                                moe_fusion_wei_addr.zp[i]->get_layout());
                        dnnl_weights[i].zp = convert2dnnl(moe_fusion_wei_addr.zp[i],
                                                          {dnnl_weights[i].ic / dnnl_weights[i].ic_group_size, dnnl_weights[i].oc},
                                                          dnnl::memory::format_tag::ab,
                                                          zp_offset);
                    }
                }
            }
        }
    }

    void init_shared_primitives(cldnn::engine& engine, const struct moe_fusion_weights_base_addr& addr, int batch) {
        if (_shared_gate_proj && _shared_gate_proj->m_batch == batch)
            return;

        OPENVINO_ASSERT(addr.shared_weight[0], "MoE shared expert enabled (num_shared_expert > 0) but shared weight buffers are not bound");

        // layout.count() is the logical element count even for sub-byte dtypes (u2/u4
        // packing only enters through bytes_count()), so this stays correct when the
        // shared weights are u2 (e.g. [512, 32, 64] u2 -> 512 with hidden 2048).
        _shared_intermediate_size = static_cast<int>(addr.shared_weight[0]->get_layout().count() / _hidden_size);
        auto eng = engine.get_onednn_engine();
        using t = onednn_matmul::type;

        // Helper: returns true if the memory is a real ZP buffer (not a dynamic placeholder used for symmetric quantization).
        auto is_valid_zp = [](const memory::ptr& m) {
            return m && m->get_layout().data_type != data_types::dynamic;
        };

        // u2 shared weights: consume the u4-unpacked scratch (filled by
        // unpack_u2_weights_for_prefill) with u4 descriptors, mirroring the
        // routed-expert grouped GEMM substitution. Non-u2 projections keep the
        // original buffers/dtypes (mixed precision).
        auto shared_weight_dt = [&](int i) {
            return _shared_weights_u2[i] ? dnnl::memory::data_type::u4 : convert_data_type(addr.shared_weight[i]->get_layout().data_type);
        };
        auto shared_weight_mem = [&](int i, const std::vector<int64_t>& dims, dnnl::memory::format_tag tag) {
            if (_shared_weights_u2[i]) {
                OPENVINO_ASSERT(_u2_unpack_shared_weight[i], "moe_3gemm shared weight ", i, " is u2 but its u4 unpack scratch is not allocated");
                return _u2_unpack_shared_weight[i]->get_onednn_memory(
                    dnnl::memory::desc(dnnl::memory::dims(dims), dnnl::memory::data_type::u4, tag));
            }
            return convert2dnnl(addr.shared_weight[i], dims, tag);
        };
        auto shared_zp_mem = [&](int i, const std::vector<int64_t>& dims, dnnl::memory::format_tag tag) {
            if (_shared_weights_u2[i]) {
                OPENVINO_ASSERT(_u2_unpack_shared_zp[i], "moe_3gemm shared weight ", i, " is u2 but its u4 zp unpack scratch is not allocated");
                return _u2_unpack_shared_zp[i]->get_onednn_memory(
                    dnnl::memory::desc(dnnl::memory::dims(dims), dnnl::memory::data_type::u4, tag));
            }
            return convert2dnnl(addr.shared_zp[i], dims, tag);
        };

        // 1. Up (Standard Linear)
        auto up_w_dt = shared_weight_dt(1);
        auto up_w = shared_weight_mem(1, {_hidden_size, _shared_intermediate_size}, dnnl::memory::format_tag::ba);
        auto up_s = addr.shared_scale[1]
                        ? convert2dnnl(addr.shared_scale[1], {_hidden_size / _gate_up_group_size, _shared_intermediate_size}, dnnl::memory::format_tag::ab)
                        : dnnl::memory();
        auto up_z = is_valid_zp(addr.shared_zp[1])
                        ? shared_zp_mem(1, {_hidden_size / _gate_up_group_size, _shared_intermediate_size}, dnnl::memory::format_tag::ab)
                        : dnnl::memory();
        _shared_up_proj = std::make_shared<onednn_linear>(onednn_linear::create(eng,
                                                                                dnnl::memory::data_type::f16,
                                                                                up_w_dt,
                                                                                batch,
                                                                                _hidden_size,
                                                                                _shared_intermediate_size,
                                                                                _gate_up_group_size,
                                                                                t::none,
                                                                                up_w,
                                                                                up_s,
                                                                                up_z));

        // 2. Gate (SiLU + BinMul)
        auto gate_w_dt = shared_weight_dt(0);
        auto gate_w = shared_weight_mem(0, {_hidden_size, _shared_intermediate_size}, dnnl::memory::format_tag::ba);
        auto gate_s = addr.shared_scale[0]
                          ? convert2dnnl(addr.shared_scale[0], {_hidden_size / _gate_up_group_size, _shared_intermediate_size}, dnnl::memory::format_tag::ab)
                          : dnnl::memory();
        auto gate_z = is_valid_zp(addr.shared_zp[0])
                          ? shared_zp_mem(0, {_hidden_size / _gate_up_group_size, _shared_intermediate_size}, dnnl::memory::format_tag::ab)
                          : dnnl::memory();
        _shared_gate_proj = std::make_shared<onednn_linear>(onednn_linear::create(eng,
                                                                                  dnnl::memory::data_type::f16,
                                                                                  gate_w_dt,
                                                                                  batch,
                                                                                  _hidden_size,
                                                                                  _shared_intermediate_size,
                                                                                  _gate_up_group_size,
                                                                                  t::with_gate_act_bin_mul,
                                                                                  gate_w,
                                                                                  gate_s,
                                                                                  gate_z,
                                                                                  moe_activation_to_dnnl_algo(_activation_type)));

        // 3. Scalar Gate (Sigmoid)
        // It is very small weight with shape of [Hidden, 1], and not need to keep compressed, so KeepMOE3GemmConstPrecision will not keep its precision and
        // ConvertPrecision will convert to f16. So Scalar gate is [Hidden, 1], f16 weights, no scale/zp for now.
        // If it stayed u2 (int2 exports), it is consumed via the u4-unpacked scratch like the other shared projections.
        dnnl::memory sg_w;
        dnnl::memory::data_type sg_w_dt = dnnl::memory::data_type::f16;
        if (addr.shared_weight[3]) {
            sg_w_dt = shared_weight_dt(3);
            sg_w = shared_weight_mem(3, {_hidden_size, 1}, dnnl::memory::format_tag::ba);
        }

        if (sg_w) {
            _shared_gate_gate_proj = std::make_shared<onednn_linear>(
                onednn_linear::
                    create(eng, dnnl::memory::data_type::f16, sg_w_dt, batch, _hidden_size, 1, -1, t::with_sigmoid, sg_w, dnnl::memory(), dnnl::memory()));
        }

        // 4. Down (BinMul + Sum)
        auto down_w_dt = shared_weight_dt(2);
        auto down_w = shared_weight_mem(2, {_shared_intermediate_size, _hidden_size}, dnnl::memory::format_tag::ba);
        auto down_s = addr.shared_scale[2]
                          ? convert2dnnl(addr.shared_scale[2], {_shared_intermediate_size / _down_group_size, _hidden_size}, dnnl::memory::format_tag::ab)
                          : dnnl::memory();
        auto down_z = is_valid_zp(addr.shared_zp[2])
                          ? shared_zp_mem(2, {_shared_intermediate_size / _down_group_size, _hidden_size}, dnnl::memory::format_tag::ab)
                          : dnnl::memory();
        _shared_down_proj = std::make_shared<onednn_linear>(onednn_linear::create(eng,
                                                                                  dnnl::memory::data_type::f16,
                                                                                  down_w_dt,
                                                                                  batch,
                                                                                  _shared_intermediate_size,
                                                                                  _hidden_size,
                                                                                  _down_group_size,
                                                                                  t::with_bin_mul_sum,
                                                                                  down_w,
                                                                                  down_s,
                                                                                  down_z));
    }

    void execute_shared_expert(dnnl::stream& stream, int batch, memory::ptr input_mem, memory::ptr output_mem, scratch_buffers& scratch) {
        if (!_shared_gate_proj || !_shared_up_proj || !_shared_down_proj || !_shared_gate_gate_proj) {
            return;
        }

        auto input_dnnl = convert2dnnl(input_mem, {batch, _hidden_size}, dnnl::memory::format_tag::ab);
        auto output_dnnl = convert2dnnl(output_mem, {batch, _hidden_size}, dnnl::memory::format_tag::ab);

        auto up_mem_dnnl = convert2dnnl(scratch.up, {batch, _shared_intermediate_size}, dnnl::memory::format_tag::ab);
        auto gate_mem_dnnl = convert2dnnl(scratch.gate, {batch, _shared_intermediate_size}, dnnl::memory::format_tag::ab);
        // Reuse routing weights or topk_weights buffer as temp for scalar gate [Batch, 1]
        // This buffer is usually [Batch, MaxTopK]. Check if big enough. Batch=1 -> TopK. Batch>1 -> Batch*TopK.
        // We only use [Batch, 1] part.
        auto scalar_gate_mem = scratch.topk_weights;
        auto scalar_gate_dnnl = convert2dnnl(scalar_gate_mem, {batch, 1}, dnnl::memory::format_tag::ab);

        // 1. Up Proj
        _shared_up_proj->forward(stream, batch, input_dnnl, up_mem_dnnl, dnnl::memory());

        // 2. Gate Proj (Fused with SiLU and BinMul(up))
        _shared_gate_proj->forward(stream, batch, input_dnnl, gate_mem_dnnl, up_mem_dnnl);

        // 3. Scalar Gate Proj (Fused with Sigmoid)
        _shared_gate_gate_proj->forward(stream, batch, input_dnnl, scalar_gate_dnnl, dnnl::memory());

        // 4. Down Proj (Fused with BinMul(Scalar) and Sum)
        _shared_down_proj->forward(stream, batch, gate_mem_dnnl, output_dnnl, scalar_gate_dnnl);
    }

    void save(BinaryOutputBuffer& ob) const override {
        PrimitiveImplOCL::save(ob);
        ob << use_micro_gemm_prefill;
        ob << use_gpu_mask_gen_prefill;
        ob << use_grouped_gemm_prefill;
    }

    void load(BinaryInputBuffer& ib) override {
        PrimitiveImplOCL::load(ib);
        // Read execution-path flags before init() so any future init() logic
        // that depends on them sees the deserialized (not default) values.
        ib >> use_micro_gemm_prefill;
        ib >> use_gpu_mask_gen_prefill;
        ib >> use_grouped_gemm_prefill;
        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        _gemm_weights_u2[0] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type == data_types::u2;
        _gemm_weights_u2[1] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1)).data_type == data_types::u2;
        _gemm_weights_u2[2] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2)).data_type == data_types::u2;
        _weights_u2 = _gemm_weights_u2[0] || _gemm_weights_u2[1] || _gemm_weights_u2[2];
        if (impl_params->input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)) {
            _shared_weights_u2[0] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT)).data_type == data_types::u2;
            _shared_weights_u2[1] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_UP_WEIGHT)).data_type == data_types::u2;
            _shared_weights_u2[2] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_DOWN_WEIGHT)).data_type == data_types::u2;
            _shared_weights_u2[3] = impl_params->get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_GATE_WEIGHT)).data_type == data_types::u2;
        }
        init(impl_params->typed_desc<moe_3gemm_fused_compressed>());
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto cur_moe = make_deep_copy<moe_3gemm_swiglu_opt_impl>(this);
        cur_moe->_dnnl_weights = _dnnl_weights;
        cur_moe->_hidden_size = _hidden_size;
        cur_moe->_intermediate_size = _intermediate_size;
        cur_moe->_gate_up_group_size = _gate_up_group_size;
        cur_moe->_down_group_size = _down_group_size;
        cur_moe->_weight_provider = _weight_provider;  // shared across clones within the same network
        cur_moe->use_micro_gemm_prefill = use_micro_gemm_prefill;
        cur_moe->use_gpu_mask_gen_prefill = use_gpu_mask_gen_prefill;
        cur_moe->use_grouped_gemm_prefill = use_grouped_gemm_prefill;
        cur_moe->_weights_u2 = _weights_u2;
        cur_moe->_gemm_weights_u2[0] = _gemm_weights_u2[0];
        cur_moe->_gemm_weights_u2[1] = _gemm_weights_u2[1];
        cur_moe->_gemm_weights_u2[2] = _gemm_weights_u2[2];
        // Don't share the u4 unpack scratch with the clone; the clone fills its own on first
        // prefill. make_deep_copy() above copied the cache keys too, so clear them as well —
        // otherwise the clone would see a "already unpacked" key next to a null buffer.
        cur_moe->_u2_unpack_weight[0] = nullptr;
        cur_moe->_u2_unpack_weight[1] = nullptr;
        cur_moe->_u2_unpack_weight[2] = nullptr;
        cur_moe->_u2_unpack_zp[0] = nullptr;
        cur_moe->_u2_unpack_zp[1] = nullptr;
        cur_moe->_u2_unpack_zp[2] = nullptr;
        for (int i = 0; i < 3; i++) {
            cur_moe->_u2_unpack_weight_key[i].reset();
            cur_moe->_u2_unpack_zp_key[i].reset();
        }
        for (int i = 0; i < 4; i++) {
            cur_moe->_u2_unpack_shared_weight_key[i].reset();
            cur_moe->_u2_unpack_shared_zp_key[i].reset();
        }
        cur_moe->_shared_weights_u2[0] = _shared_weights_u2[0];
        cur_moe->_shared_weights_u2[1] = _shared_weights_u2[1];
        cur_moe->_shared_weights_u2[2] = _shared_weights_u2[2];
        cur_moe->_shared_weights_u2[3] = _shared_weights_u2[3];
        cur_moe->_u2_unpack_shared_weight[0] = nullptr;
        cur_moe->_u2_unpack_shared_weight[1] = nullptr;
        cur_moe->_u2_unpack_shared_weight[2] = nullptr;
        cur_moe->_u2_unpack_shared_weight[3] = nullptr;
        cur_moe->_u2_unpack_shared_zp[0] = nullptr;
        cur_moe->_u2_unpack_shared_zp[1] = nullptr;
        cur_moe->_u2_unpack_shared_zp[2] = nullptr;
        cur_moe->_u2_unpack_shared_zp[3] = nullptr;
        cur_moe->batched_gemv_threshold = batched_gemv_threshold;
        cur_moe->_activation_type = _activation_type;
        return cur_moe;
    }

    // Notice: don't change the order of internal buffers, it is defined in MOE3GemmInternalBufferIdx
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_moe = params.typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        size_t max_topk = static_cast<size_t>(config.top_k);
        size_t expert_num = static_cast<size_t>(config.num_expert);
        auto hidden_states_layout = params.input_layouts[0];
        auto token_num = get_seq_len(hidden_states_layout);
        auto data_type = hidden_states_layout.data_type;
        bool has_shared_expert = params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT);

        std::vector<BufferDescriptor> internal_buffers;
        // To support micro_gemm, prefill need to allocate max_topk * token_num for input data of micro_gemm
        auto max_batch = has_shared_expert ? (max_topk + 1) * token_num : max_topk * token_num;
        layout layout_gateup_out(ov::Shape{max_batch, static_cast<size_t>(config.inter_size)}, data_type, cldnn::format::bfyx);
        layout layout_down_out(ov::Shape{max_batch, static_cast<size_t>(config.hidden_size)}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(layout_gateup_out, false);  // 0: up output (GPU-only)
        internal_buffers.emplace_back(layout_down_out, false);    // 1: down output (GPU-only)
        // onednn: scratch.x, scratch.routing_weights = gather(x, ...)
        //         scratch.up = up(scratch.x)
        //         scratch.gate = gate(scratch.x) * scratch.up
        //         scratch.y = down(scratch.gate) * routing_weights
        internal_buffers.emplace_back(layout_down_out, false);  // 2: up/gate input, scratch.x has same layout with down output (GPU-only)
        layout routing_layout(ov::Shape{max_batch}, data_type, cldnn::format::bfyx);
        internal_buffers.emplace_back(routing_layout, true);      // 3: routing_weights
        internal_buffers.emplace_back(layout_gateup_out, false);  // 4: gate output, scratch.gate has same layout with up (GPU-only)

        // expert masks for gpu
        layout index_layout(ov::Shape{expert_num, token_num}, ov::element::i32, cldnn::format::bfyx);
        internal_buffers.emplace_back(index_layout, true);  // 5: expert_mask_batch
        internal_buffers.emplace_back(index_layout, true);  // 6: expert_mask_topk

        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] get_internal_buffer_descs(): use_micro_gemm_prefill=" << use_micro_gemm_prefill
                               << ", use_grouped_gemm_prefill=" << use_grouped_gemm_prefill << std::endl;
        // for micro_gemm
        if (use_micro_gemm_prefill && token_num > 1) {
            layout layout_micro_gemm(ov::Shape{expert_num, token_num}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 7: experts_ids for each activated expert
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 8: token start offset idx (input gather tokens) for each activated expert
            internal_buffers.emplace_back(layout_micro_gemm, true);  // 9: token len (input gather tokens) for each activated expert
            layout layout_token_idx(ov::Shape{token_num * max_topk}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_token_idx, true);  // 10: token idx per expert
            layout layout_actual_used_expert_num(ov::Shape{1}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_actual_used_expert_num, false);  // 11: actual_used_expert_num
        }
        // for grouped_gemm: shared metadata buffers (7-11) + int32_t expert-row-offsets (12)
        if (use_grouped_gemm_prefill && token_num > 1) {
            layout layout_meta(ov::Shape{expert_num, token_num}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_meta, true);  // 7: activated expert ids
            internal_buffers.emplace_back(layout_meta, true);  // 8: token start offset per activated expert
            internal_buffers.emplace_back(layout_meta, true);  // 9: token len per activated expert
            layout layout_token_idx(ov::Shape{token_num * max_topk}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_token_idx, true);  // 10: flat token idx per expert (for gather)
            layout layout_actual_used_expert_num(ov::Shape{1}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_actual_used_expert_num, false);  // 11: actual_used_expert_num
            // int32_t end-offsets per expert for OneDNN grouped memory descriptor
            // offsets[e] = sum(n_0..n_e), the exclusive end index of expert e in the flat buffer
            layout layout_grouped_offsets(ov::Shape{expert_num}, ov::element::i32, cldnn::format::bfyx);
            internal_buffers.emplace_back(layout_grouped_offsets, true);  // 12: grouped end-offsets
        }
        return internal_buffers;
    }

    void prepare_internal_buffers(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, scratch_buffers& scratch, size_t token_num) {
        const auto& intermediates_memories = instance.get_intermediates_memories();
        auto& engine = instance.get_network().get_engine();

        // topk_id / topk_weights are read from inputs (computed by MoERouterFused).
        scratch.topk_weights = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::TOPK_WEIGHTS));
        scratch.topk_id = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::TOPK_INDICES));
        scratch.up = intermediates_memories[MOE_INTERNAL_BUFFER_UP_OUTPUT];
        scratch.y = intermediates_memories[MOE_INTERNAL_BUFFER_DOWN_OUTPUT];
        // Routing weights scratch buffer (used in prefill paths and reused as shared_gate_vals in batched GEMV)
        scratch.routing_weights = intermediates_memories[MOE_INTERNAL_BUFFER_ROUTING_WEIGHTS];
        if (token_num > 1) {
            scratch.x = intermediates_memories[MOE_INTERNAL_BUFFER_GATE_UP_INPUT];
            scratch.gate = intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT];
            const auto& config = instance.get_typed_desc<moe_3gemm_fused_compressed>()->_config;
            int expert_num = static_cast<int>(config.num_expert);
            scratch.expert_masks.resize(expert_num);
            for (int i = 0; i < expert_num; i++) {
                auto mask_layout = cldnn::layout({static_cast<int>(token_num)}, cldnn::data_types::i32, cldnn::format::get_default_format(1));
                scratch.expert_masks[i].batch =
                    engine.create_subbuffer(*intermediates_memories[MOE_INTERNAL_BUFFER_EXPERT_MASK_BATCH], mask_layout, i * token_num * sizeof(int32_t));
                scratch.expert_masks[i].topk =
                    engine.create_subbuffer(*intermediates_memories[MOE_INTERNAL_BUFFER_EXPERT_MASK_TOPK], mask_layout, i * token_num * sizeof(int32_t));
            }
        }

        prepare_weight_pointers(instance, scratch);
    }

    void get_expert_mask_from_gpu(const MOECompressed::Config& config, memory::ptr mem, stream& stream, expert_mask_cpu& expert_mask, size_t actual_token_num) {
        // shape: [token_num, topk]
        auto layout = mem->get_layout();

        int max_expert_num = static_cast<int>(config.num_expert);
        int max_topk = static_cast<int>(config.top_k);
        int max_tokens = static_cast<int>(actual_token_num);

        expert_mask.pred_flag.resize(max_expert_num, 0);
        expert_mask.batch.resize(max_expert_num, {});
        expert_mask.topk.resize(max_expert_num, {});

        GPU_DEBUG_TRACE_DETAIL << "[DEBUG] get_expert_mask_from_gpu: max_expert_num=" << max_expert_num << ", max_topk=" << max_topk
                               << ", max_tokens=" << max_tokens << std::endl;
        std::vector<int32_t> buf(max_topk * max_tokens);
        mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

        for (int b = 0; b < max_tokens; b++) {
            auto* tok_p = &buf[b * max_topk];
            for (int t = 0; t < max_topk; t++) {
                auto expert_no = tok_p[t];
                if (expert_no >= max_expert_num) {
                    OPENVINO_THROW("expert_no ", expert_no, " exceed max_expert_num ", max_expert_num);
                }

                expert_mask.batch[expert_no].push_back(b);
                expert_mask.topk[expert_no].push_back(t + b * max_topk);
                expert_mask.pred_flag[expert_no] = 1;
            }
        }
        {
            // check if the result is ok
            int count = 0;
            for (int no = 0; no < max_expert_num; no++) {
                count += static_cast<int>(expert_mask.batch[no].size());
            }
            OPENVINO_ASSERT(count == max_topk * max_tokens,
                            "With max_expert_num=",
                            max_expert_num,
                            ",max_topk=",
                            max_topk,
                            ",max_tokens=",
                            max_tokens,
                            " should have ",
                            max_topk * max_tokens,
                            " tokens, but current is ",
                            count);
        }
    }

    void copy_expert_mask_to_gpu(stream& stream, const expert_mask_cpu& expert_mask, size_t expert_no, expert_mask_gpu& expert_mask_mem) {
        auto size = expert_mask.batch[expert_no].size() * sizeof(int);
        expert_mask_mem.batch->copy_from(stream, expert_mask.batch[expert_no].data(), 0, 0, size, true);
        expert_mask_mem.topk->copy_from(stream, expert_mask.topk[expert_no].data(), 0, 0, size, true);
    }

    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<memory::ptr> inputs,
                                    std::vector<memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local,
                                    bool needs_completion_event = false,
                                    std::vector<int> scalar_inputs = {}) const {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("moe_3gemm_swiglu_opt_impl::execute_stage"));
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;

        GPU_DEBUG_TRACE_DETAIL << "moe::execute_stage: " << stage.kernel->get_id() << std::endl;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
            GPU_DEBUG_TRACE_DETAIL << "\tinput[" << i << "]: " << inputs[i]->get_layout().to_short_string() << std::endl;
        }

        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
            GPU_DEBUG_TRACE_DETAIL << "\toutput[" << i << "]: " << outputs[i]->get_layout().to_short_string() << std::endl;
        }

        cldnn::scalars_desc scalar_desc;
        if (!scalar_inputs.empty()) {
            scalar_desc.resize(scalar_inputs.size());
            for (uint32_t i = 0; i < scalar_inputs.size(); i++) {
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
                scalar_desc[i].t = ScalarDescriptor::Types::INT32;
                scalar_desc[i].v.s32 = scalar_inputs[i];
            }
            args.scalars = &scalar_desc;
            GPU_DEBUG_TRACE_DETAIL << "\tscalar_inputs: ";
            for (const auto& scalar : scalar_inputs) {
                GPU_DEBUG_TRACE_DETAIL << scalar << " ";
            }
            GPU_DEBUG_TRACE_DETAIL << std::endl;
        }

        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;

        if (global.size() == 2) {
            GPU_DEBUG_TRACE_DETAIL << "\tgws = {" << global[0] << ", " << global[1] << "}" << std::endl;
            GPU_DEBUG_TRACE_DETAIL << "\tlws = {" << local[0] << ", " << local[1] << "}" << std::endl;
        } else if (global.size() == 3) {
            GPU_DEBUG_TRACE_DETAIL << "\tgws = {" << global[0] << ", " << global[1] << ", " << global[2] << "}" << std::endl;
            GPU_DEBUG_TRACE_DETAIL << "\tlws = {" << local[0] << ", " << local[1] << ", " << local[2] << "}" << std::endl;
        }

        kernel_dump_info.add_entry_point(stage.kernel->get_id());

        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, needs_completion_event);
    }

    auto get_input_info(typed_primitive_inst<moe_3gemm_fused_compressed>& instance, int idx) {
        auto mem = instance.input_memory_ptr(idx);
        auto dep = instance.dependencies()[idx];
        auto layout = dep.first->get_impl_params()->get_output_layout(dep.second);
        return std::make_tuple(mem, layout);
    }

    // Batched GEMV path: handles token_num >= 1 with optimized GEMV kernels.
    // Each workgroup processes one (token, expert) pair. Avoids gather/scatter/CPU-sync overhead of prefill paths.
    // Supports shared expert: EXPERTS_PER_TOKEN = MAX_TOPK + 1 when shared expert is enabled.
    cldnn::event::ptr exec_batched_gemv(const std::vector<cldnn::event::ptr>& events,
                                        typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                        scratch_buffers& scratch,
                                        size_t token_num) {
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        int max_topk = static_cast<int>(cur_moe->_config.top_k);

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto batch_mem_ptr = scratch.topk_id;
        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto routing_mem_ptr = scratch.topk_weights;
        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);

        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;
        const size_t max_work_group_size = instance.get_impl_params()->get_device_info().max_work_group_size;

        {
            const size_t topk_count = token_num * static_cast<size_t>(max_topk);
            bool needs_fallback = false;
            on_before_batched_gemv(stream, instance, scratch, topk_count, needs_fallback);
            if (needs_fallback) {
                // Cannot fit all experts simultaneously → fall back to per-expert onednn loop
                instance.output_memory_ptr(0)->fill(stream, 0u);
                return exec_prefill_onednn(events, stream, instance, scratch);
            }
        }
        // OTD hook may have remapped expert IDs into scratch._expert_index_buffer
        if (scratch._expert_index_buffer) {
            batch_mem_ptr = scratch._expert_index_buffer;
        }

        // gate
        const auto& mlp_gate_wei_mem = scratch.moe_fusion_wei_addr.weight[0];
        const auto& mlp_gate_scale_mem = scratch.moe_fusion_wei_addr.scale[0];
        const auto& mlp_gate_zp_mem = scratch.moe_fusion_wei_addr.zp[0];

        // up
        const auto& mlp_up_wei_mem = scratch.moe_fusion_wei_addr.weight[1];
        const auto& mlp_up_scale_mem = scratch.moe_fusion_wei_addr.scale[1];
        const auto& mlp_up_zp_mem = scratch.moe_fusion_wei_addr.zp[1];

        // down
        const auto& mlp_down_wei_mem = scratch.moe_fusion_wei_addr.weight[2];
        const auto& mlp_down_scale_mem = scratch.moe_fusion_wei_addr.scale[2];
        const auto& mlp_down_zp_mem = scratch.moe_fusion_wei_addr.zp[2];
        event::ptr ret;

        size_t compute_experts = static_cast<size_t>(max_topk);
        Stage* stage_gate_up = mlp_gate_up.get();
        Stage* stage_down = mlp_down.get();
        Stage* stage_reduce = mlp_reduce.get();

        std::vector<memory::ptr> extra_args_gate_up;
        std::vector<memory::ptr> extra_args_down;

        if (_has_shared_expert) {
            compute_experts += 1;

            extra_args_gate_up = {scratch.moe_fusion_wei_addr.shared_weight[0],
                                  scratch.moe_fusion_wei_addr.shared_scale[0],
                                  scratch.moe_fusion_wei_addr.shared_zp[0],
                                  scratch.moe_fusion_wei_addr.shared_weight[1],
                                  scratch.moe_fusion_wei_addr.shared_scale[1],
                                  scratch.moe_fusion_wei_addr.shared_zp[1],
                                  scratch.moe_fusion_wei_addr.shared_weight[3],
                                  scratch.routing_weights};  // reused as shared_gate_out [token_num]

            extra_args_down = {scratch.moe_fusion_wei_addr.shared_weight[2],
                               scratch.moe_fusion_wei_addr.shared_scale[2],
                               scratch.moe_fusion_wei_addr.shared_zp[2]};
        }

        GPU_DEBUG_TRACE_DETAIL << "\nexec_batched_gemv(): token_num=" << token_num << ", max_topk=" << max_topk << ", has_shared=" << _has_shared_expert
                               << std::endl;

        {
            // scratch.up = up(x) * silu(gate(x)) for all (token, expert) pairs
            std::vector<memory::ptr> args_gate_up =
                {batch_mem_ptr, mlp_gate_wei_mem, mlp_gate_scale_mem, mlp_gate_zp_mem, mlp_up_wei_mem, mlp_up_scale_mem, mlp_up_zp_mem};
            if (_has_shared_expert)
                args_gate_up.insert(args_gate_up.end(), extra_args_gate_up.begin(), extra_args_gate_up.end());
            args_gate_up.push_back(hidden_states_mem_ptr);

            auto ret_event = execute_stage(events,
                                           instance,
                                           *stage_gate_up,
                                           args_gate_up,
                                           {scratch.up},
                                           {token_num * compute_experts, subgroup_size, static_cast<size_t>(_intermediate_size / N_BLOCK)},
                                           {1, subgroup_size, SUBGROUP_NUM});

            // scratch.y = down(scratch.up) * routing_weight for all (token, expert) pairs
            std::vector<memory::ptr> args_down = {batch_mem_ptr, mlp_down_wei_mem, mlp_down_scale_mem, mlp_down_zp_mem};
            if (_has_shared_expert)
                args_down.insert(args_down.end(), extra_args_down.begin(), extra_args_down.end());
            args_down.push_back(scratch.up);
            args_down.push_back(routing_mem_ptr);  // compact topk_weights [token_num * MAX_TOPK]
            if (_has_shared_expert)
                args_down.push_back(scratch.routing_weights);  // shared_gate_in [token_num]
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *stage_down,
                                      args_down,
                                      {scratch.y},
                                      {token_num * compute_experts, subgroup_size, static_cast<size_t>(_hidden_size / N_BLOCK)},
                                      {1, subgroup_size, SUBGROUP_NUM});

            // Per-token reduction: final[t] = sum(scratch.y[t * REDUCE_COUNT .. (t+1) * REDUCE_COUNT - 1])
            ret = execute_stage({ret_event},
                                instance,
                                *stage_reduce,
                                {scratch.y},
                                {final_hidden_states_mem_ptr},
                                {token_num, static_cast<size_t>(_hidden_size)},
                                {1, std::min(max_work_group_size, size_t{1024})},
                                instance.needs_completion_event());
        }
        return ret;
    }

    cldnn::event::ptr exec_prefill_micro_gemm(const std::vector<cldnn::event::ptr>& events,
                                              typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                              scratch_buffers& scratch,
                                              const bool use_gpu_mask_gen) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        int max_topk = static_cast<int>(cur_moe->_config.top_k);
        const auto& config = cur_moe->_config;

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        // [batch, max_topk]
        auto batch_mem_ptr = scratch.topk_id;
        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto routing_mem_ptr = scratch.topk_weights;
        auto input_layout = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES))->get_layout();
        auto token_num = get_seq_len(input_layout);

        _hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
        _intermediate_size = static_cast<int>(cur_moe->_config.inter_size);

        auto rtp = static_cast<MoE3GemmRuntimeParams*>(m_rt_params.get());
        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;

        event::ptr ret_event;
        const auto& intermediates_memories = instance.get_intermediates_memories();
        auto& stream = instance.get_network().get_stream();
        auto num_total_experts = static_cast<int>(cur_moe->_config.num_expert);
        int num_actually_used_experts = 0;

        {
            auto topk_count = token_num * static_cast<size_t>(max_topk);
            bool needs_fallback = false;
            on_before_prefill(stream, instance, scratch, batch_mem_ptr, topk_count, needs_fallback);
            if (needs_fallback) {
                instance.output_memory_ptr(0)->fill(stream, 0u);
                return exec_prefill_onednn(events, stream, instance, scratch);
            }
        }

        // step 1: generate 4 mask data for following kernel execution
        // input: topk output, [token_len, expert_topk]
        // output:
        //   mask 0: token idx per expert, flat array of length token_len * expert_topk
        //             (experts are laid out consecutively; use experts_info_start_idx + tokens_lens_per_expert to slice)
        //   mask 1: token start offset idx in mask 0 for each activated expert, shape = [activated_expert_num]
        //   mask 2: token len for each activated expert, shape = [activated_expert_num]
        //   mask 3: expert id, shape = [activated_expert_num]
        //   mask 4: actual activated expert num, shape = [1]
        if (use_gpu_mask_gen) {
            auto token_size = token_num;
            ret_event = execute_stage(events,
                                      instance,
                                      *prefill_mask_gen,
                                      {batch_mem_ptr},
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]},
                                      {static_cast<size_t>(num_total_experts), 1, 1},
                                      {static_cast<size_t>(num_total_experts), 1, 1},
                                      false,
                                      {static_cast<int>(token_size)});

            // num_actually_used_experts is needed for micro_gem wgs, need sync
            ret_event->wait();
            cldnn::mem_lock<int32_t, mem_lock_type::read> num_actual_experts_lock(intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM], stream);
            rtp->num_actually_used_experts = num_actual_experts_lock[0];
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "Step 1: mask gen by gpu, num_actually_used_experts = " << rtp->num_actually_used_experts << std::endl;
#    endif
        } else {
            ret_event = events.empty() ? nullptr : events[0];
            expert_mask_cpu expert_mask_cpu;
            get_expert_mask_from_gpu(config, batch_mem_ptr, stream, expert_mask_cpu, token_num);

            auto token_size = token_num;
            auto max_topk = static_cast<int>(cur_moe->_config.top_k);
            std::vector<int32_t> tokens_per_expert_cpu(token_size * max_topk, -1);
            std::vector<int32_t> tokens_lens_per_expert_cpu(num_total_experts, -1);
            std::vector<int32_t> experts_info_start_idx_cpu(num_total_experts, -1);
            std::vector<int32_t> experts_id_cpu(num_total_experts, -1);

            int tokens_per_expert_iter = 0;
            int experts_id_iter = 0;
            for (int expert_idx = 0; expert_idx < num_total_experts; expert_idx++) {
                if (!expert_mask_cpu.batch[expert_idx].empty()) {
                    experts_info_start_idx_cpu[experts_id_iter] = tokens_per_expert_iter;
                    experts_id_cpu[experts_id_iter] = expert_idx;
                    tokens_lens_per_expert_cpu[experts_id_iter++] = static_cast<int32_t>(expert_mask_cpu.batch[expert_idx].size());
                    num_actually_used_experts++;
                    for (auto t : expert_mask_cpu.batch[expert_idx]) {
                        tokens_per_expert_cpu[tokens_per_expert_iter++] = t;
                    }
                }
            }
            rtp->num_actually_used_experts = num_actually_used_experts;

            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]
                ->copy_from(stream, tokens_per_expert_cpu.data(), 0, 0, tokens_per_expert_cpu.size() * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT]
                ->copy_from(stream, experts_info_start_idx_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS]
                ->copy_from(stream, experts_id_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);
            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT]
                ->copy_from(stream, tokens_lens_per_expert_cpu.data(), 0, 0, num_actually_used_experts * sizeof(int32_t), true);

            intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]->copy_from(stream, &num_actually_used_experts, 0, 0, sizeof(int32_t), true);

#    if DEBUG_MOE_LOG
            {
                GPU_DEBUG_TRACE_DETAIL << "\nstep 1: prefill_mask num_actually_used_experts=" << num_actually_used_experts << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "expert_id[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << experts_id_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "experts_info_start_idx[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << experts_info_start_idx_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "tokens_len_per_expert[" << num_actually_used_experts << "]: = ";
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << tokens_lens_per_expert_cpu[i] << ", ";
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
                GPU_DEBUG_TRACE_DETAIL << "tokens_per_expert[" << num_actually_used_experts << "]:" << std::endl;
                int token_idx = 0;
                for (int i = 0; i < num_actually_used_experts; i++) {
                    GPU_DEBUG_TRACE_DETAIL << "\texpert[" << i << "]: = ";
                    for (int j = 0; j < tokens_lens_per_expert_cpu[i]; j++) {
                        GPU_DEBUG_TRACE_DETAIL << tokens_per_expert_cpu[token_idx + j] << ", ";
                    }
                    token_idx += tokens_lens_per_expert_cpu[i];
                    GPU_DEBUG_TRACE_DETAIL << std::endl;
                }
                GPU_DEBUG_TRACE_DETAIL << std::endl;
            }
#    endif
        }

        // step 2: generate gather input tokens
        //  input
        //      0: input tensor, shape = [token_len, hidden_size]
        //      1: token idx per expert, static shape = [token_num * topK_num]
        //  output
        //      0: gathered token: shape = [token_len * expert_topK, hidden_size]
        {
            auto hidden_size = _hidden_size;
            auto block_size = get_vec_size(*instance.get_impl_params());
            auto [local_threads_count, batches_per_thread, unaligned_elements] =
                calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), block_size, hidden_size);
            auto token_per_expert = intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]->get_layout().get_shape()[0];

#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 2: prefill_gather local_threads_count=" << local_threads_count << ", batches_per_thread=" << batches_per_thread
                                   << ", unaligned_elements=" << unaligned_elements << ", token_per_expert=" << token_per_expert
                                   << ", block_size = " << block_size << std::endl;
#    endif
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_gather,
                                      {instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)),
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]},
                                      {scratch.x},
                                      {static_cast<size_t>(token_per_expert * local_threads_count), 1, 1},
                                      {static_cast<size_t>(local_threads_count), 1, 1});
        }

        // step 3: moe_gemm for up and gate
        //  input
        //      0: gathered token, shape = [token_len * expert_topK, hidden_size]
        //      1: moe weights
        //      2: expert id, dynamic shape = [activated_expert_num]
        //      3: token start offset idx (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      4: token len (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      5: m = itermedia_size
        //      6: k = hidden_size
        //      7: wei_scale
        //      8: wei_zp
        //  output:
        //      0: up/gate output, shape = [token_len * expert_topK, hidden_size]
        // Note: If POST_PROC_SILU_MUL is enabled, silu_mul result will be involved in micro_gemm_gate, don't change kernel executor order.
        {
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 3: moe_gemm for up and gate" << std::endl;
#    endif
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_up);
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_gate);
        }

        // step 4: post proc - gate_up = silu(gate)*up, silu(x)=x*sigmod(x)=x*(1+exp(-x))
        //  input
        //      0: up  [token_len * expert_topK, hidden_size]
        //      1: gate  [token_len * expert_topK, hidden_size]
        // output
        //      0: gate_up  [token_len * expert_topK, hidden_size]
        // Note: If POST_PROC_SILU_MUL is disabled, single silu_mul kernel will be submmited.
        //       Otherwise, silu_mul has been involved in micro_gemm_gate kernel, skip here.
        const bool enable_silu_mul = ENABLE_MOE_MICRO_GEMM_POST_PROC_SILU_MUL;
        if (!enable_silu_mul) {
            auto token_size = token_num * max_topk;
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 4: prefill_swiglu token_size=" << token_size << ", hidden_size=" << _intermediate_size << std::endl;
#    endif
            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_swiglu,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_UP_OUTPUT], intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {static_cast<size_t>(_intermediate_size), static_cast<size_t>(token_size), 1},
                                      {subgroup_size, 1, 1});
        }

        // step 5: moe_gemm for down
        //  input
        //      0: gate_up, shape = [token_len * expert_topK, hidden_size]
        //      1: moe weights
        //      2: expert id, dynamic shape = [activated_expert_num]
        //      3: token start offset idx (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      4: token len (input gather tokens) for each activated expert, dynamic shape = [activated_expert_num]
        //      5: m = itermedia_size
        //      6: k = hidden_size
        //      7: wei_scale
        //      8: wei_zp
        //  output:
        //      0: down output, shape = [token_len * expert_topK, hidden_size]
        {
#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 5: moe_gemm for down" << std::endl;
#    endif
            ret_event = PrimitiveImplOCL::execute_stage({ret_event}, instance, micro_gemm_down);
        }

        // step 6: scatter and reduce
        // input:
        //      0: down output, shape = [token_len * expert_topK, hidden_size]
        //      1: experts_per_token, shape = [token_len, expert_topK]
        //      2: expert_weights, shape = [expert_num]
        //      3: tokens_per_expert, shape = [expert_num, ?] = [token_len * expert_topK]
        //      4: experts_start_offset, shape = [activated_expert_num]
        //      5: tokens_len_per_expert,dynamic shape = [activated_expert_num]
        //      6: expert id, dynamic shape = [activated_expert_num]
        // output:
        //      0: final hidden states, shape = [token_len, hidden_size]
        {
            auto token_size = token_num;
            auto [local_threads_count, batches_per_thread, _] = calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), 4, _hidden_size);

#    if DEBUG_MOE_LOG
            GPU_DEBUG_TRACE_DETAIL << "\nstep 6: prefill_scatter_reduce token_size=" << token_size << ", local_threads_count=" << local_threads_count
                                   << ", num_actually_used_experts = " << num_actually_used_experts << std::endl;
#    endif

            ret_event = execute_stage({ret_event},
                                      instance,
                                      *prefill_scatter_reduce,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_DOWN_OUTPUT],
                                       batch_mem_ptr,
                                       routing_mem_ptr,
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]},
                                      {final_hidden_states_mem_ptr},
                                      {static_cast<size_t>(token_size * local_threads_count), 1, 1},
                                      {local_threads_count, 1, 1},
                                      true /*instance.needs_completion_event()*/);
        }

        return ret_event;
    }

    void update_rt_params(const primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoE3GemmRuntimeParams>();
        }
        update_stages_flags(instance);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        inst.update_shape_info_tensor(impl_params);
        update_rt_params(inst);
    }

    struct onednn_kernel {
        onednn_linear up;
        onednn_linear gate;
        onednn_linear down;
    };
    struct PairHash {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            // Combine hash values of the pair elements
            return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
        }
    };

    using lru_cache_hash = LruCache<std::pair<int, int>, std::shared_ptr<onednn_kernel>, PairHash>;
    lru_cache_hash _kernels = lru_cache_hash(1024);

    // --- grouped GEMM kernel cache (one primitive set per total-token count) ---
    struct grouped_onednn_kernel {
        dnnl::matmul gate_prim;
        dnnl::matmul up_prim;
        dnnl::matmul down_prim;
        dnnl::matmul::primitive_desc gate_pd;
        dnnl::matmul::primitive_desc up_pd;
        dnnl::matmul::primitive_desc down_pd;
        dnnl::memory::desc gate_scale_md;
        dnnl::memory::desc up_scale_md;
        dnnl::memory::desc down_scale_md;
        dnnl::memory::desc gate_zp_md;
        dnnl::memory::desc up_zp_md;
        dnnl::memory::desc down_zp_md;
        bool has_zp = false;
    };
    using grouped_kernel_lru = LruCache<int, std::shared_ptr<grouped_onednn_kernel>>;
    grouped_kernel_lru _grouped_kernels{128};
    onednn_kernel& get_kernel(int n_token, int expert_no, typed_primitive_inst<moe_3gemm_fused_compressed>& instance) {
        // OTD: all slots have identical shape → cache by n_token only.
        // Non-OTD: cache by (n_token, expert_no) since each expert has fixed weight handles.
        if (_weight_provider->is_offloaded()) {
            if (!_kernels.has(std::make_pair(n_token, 0))) {
                auto kernel = create_kernel(n_token, expert_no, instance);
                _kernels.add(std::make_pair(n_token, 0), kernel);
                if (auto* perf = moe_otd::get_perf_counters())
                    perf->created_onednn_kernels.fetch_add(1, std::memory_order_relaxed);
            }
            auto& kernel = *_kernels.get(std::make_pair(n_token, 0));
            // Patch weight memory handles for the current expert's LRU slot.
            // on_load_expert_weights already updated _dnnl_weights[expert_no] with
            // the correct slot offset, so copy those handles to the cached kernel.
            auto& dw = _dnnl_weights[expert_no];
            kernel.gate.weight = dw[0].weight;
            kernel.gate.scale = dw[0].scale;
            kernel.gate.zp = dw[0].zp;
            kernel.up.weight = dw[1].weight;
            kernel.up.scale = dw[1].scale;
            kernel.up.zp = dw[1].zp;
            kernel.down.weight = dw[2].weight;
            kernel.down.scale = dw[2].scale;
            kernel.down.zp = dw[2].zp;
            return kernel;
        }
        auto key = std::make_pair(n_token, expert_no);
        if (!_kernels.has(key)) {
            auto kernel = create_kernel(n_token, expert_no, instance);
            _kernels.add(key, kernel);
            if (auto* perf = moe_otd::get_perf_counters())
                perf->created_onednn_kernels.fetch_add(1, std::memory_order_relaxed);
        }
        return *_kernels.get(key);
    }

    std::shared_ptr<onednn_kernel> create_kernel(int n_token, int expert_no, typed_primitive_inst<moe_3gemm_fused_compressed>& instance) {
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();
        auto& dnn_stream = stream.get_onednn_stream();
        auto hidden_states_layout_dt =
            convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES))->get_layout().data_type);

        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto gate_activation_algo = moe_activation_to_dnnl_algo(cur_moe->_config.activation_type);

        auto& dnnl_weights = _dnnl_weights[expert_no];
        auto kernel = std::make_shared<onednn_kernel>();

        // gate
        auto gate_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0))->get_layout().data_type);
        kernel->gate = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             gate_weight_layout_dt,
                                             n_token,
                                             dnnl_weights[0].ic,
                                             dnnl_weights[0].oc,
                                             dnnl_weights[0].ic_group_size,
                                             onednn_matmul::type::with_gate_act_bin_mul,
                                             dnnl_weights[0].weight,
                                             dnnl_weights[0].scale,
                                             dnnl_weights[0].zp,
                                             gate_activation_algo);

        // up
        auto up_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_1))->get_layout().data_type);
        kernel->up = onednn_linear::create(dnn_stream.get_engine(),
                                           hidden_states_layout_dt,
                                           up_weight_layout_dt,
                                           n_token,
                                           dnnl_weights[1].ic,
                                           dnnl_weights[1].oc,
                                           dnnl_weights[1].ic_group_size,
                                           onednn_matmul::type::none,
                                           dnnl_weights[1].weight,
                                           dnnl_weights[1].scale,
                                           dnnl_weights[1].zp);

        // down
        auto down_weight_layout_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_2))->get_layout().data_type);
        kernel->down = onednn_linear::create(dnn_stream.get_engine(),
                                             hidden_states_layout_dt,
                                             down_weight_layout_dt,
                                             n_token,
                                             dnnl_weights[2].ic,
                                             dnnl_weights[2].oc,
                                             dnnl_weights[2].ic_group_size,
                                             onednn_matmul::type::with_bin_mul_per_row,
                                             dnnl_weights[2].weight,
                                             dnnl_weights[2].scale,
                                             dnnl_weights[2].zp);
        return kernel;
    }

    // Build (and cache) three grouped dnnl::matmul primitives for gate/up/down.
    // Cache key is total_tokens only — the per-request max_tokens_per_expert
    // dispatch hint is passed as a runtime argument (DNNL_ARG_HINT_MAX_GROUP_SIZE)
    // at execute() time, so no recompilation is needed when it changes.
    grouped_onednn_kernel& get_grouped_kernel(int total_tokens, typed_primitive_inst<moe_3gemm_fused_compressed>& instance) {
        auto key = total_tokens;
        if (_grouped_kernels.has(key)) {
            return *_grouped_kernels.get(key);
        }

        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& engine = instance.get_network().get_engine();
        auto& onednn_engine = engine.get_onednn_engine();

        // In OTD mode, weight buffer holds only resident_slot_count() slots, not full num_expert.
        int num_experts = get_num_grouped_experts(static_cast<int>(config.num_expert));
        auto a_dt = convert_data_type(instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES))->get_layout().data_type);
        // u2 GEMMs are unpacked into u4 scratch buffers before the grouped GEMM runs,
        // so their weight/ZP descriptors are created with the u4 dtype.
        auto grouped_weight_dt = [&](MOE3GemmInputIndex idx) {
            auto dt = instance.input_memory_ptr(static_cast<size_t>(idx))->get_layout().data_type;
            return dt == data_types::u2 ? dnnl::memory::data_type::u4 : convert_data_type(dt);
        };
        auto gw_dt = grouped_weight_dt(MOE3GemmInputIndex::WEIGHT_0);
        auto uw_dt = grouped_weight_dt(MOE3GemmInputIndex::WEIGHT_1);
        auto dw_dt = grouped_weight_dt(MOE3GemmInputIndex::WEIGHT_2);

        // Use the model config to determine ZP presence (symmetric vs asymmetric quantization)
        bool has_zp = config.has_zp;

        int K_gu = _hidden_size;        // K for gate / up
        int N_gu = _intermediate_size;  // N for gate / up
        int K_d = _intermediate_size;   // K for down
        int N_d = _hidden_size;         // N for down

        // Helper: create one grouped matmul prim-desc [total_tokens, K]*W[E,K,N]->[total_tokens,N]
        // Weights layout in memory is [E, N, K] (stored transposed), expressed as acb over dims {E,K,N}.
        auto make_pd = [&](int K, int N, int group_size, dnnl::memory::data_type w_dt) {
            dnnl::primitive_attr attr;
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

            bool has_k_groups = (group_size < K);
            if (has_k_groups) {
                // per-expert(0) x per-K-group(1) x per-N-channel(2)
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {group_size, 1}, dnnl::memory::data_type::f16);
                if (has_zp) {
                    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {group_size, 1}, w_dt);
                }
            } else {
                // per-expert(0) x per-N-channel(2), no K-grouping
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, dnnl::memory::data_type::f16);
                if (has_zp) {
                    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, w_dt);
                }
            }

            // Grouped src/dst: tokens are grouped by expert along axis-0
            auto src_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, K}, a_dt, 0, num_experts, dnnl::memory::data_type::s32);
            auto dst_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, N}, a_dt, 0, num_experts, dnnl::memory::data_type::s32);
            // Weight: logical [E, K, N], physical layout acb -> stored as [E, N, K]
            auto w_md = dnnl::memory::desc(dnnl::memory::dims{num_experts, K, N}, w_dt, dnnl::memory::format_tag::acb);

            return dnnl::matmul::primitive_desc(onednn_engine, src_md, w_md, dst_md, attr);
        };

        // Helper: create scale/ZP memory descriptor for grouped weights
        auto make_quant_md = [&](int E, int K, int group_size, int N, dnnl::memory::data_type dt) {
            int num_k_groups = K / group_size;
            if (num_k_groups > 1) {
                return dnnl::memory::desc({E, num_k_groups, N}, dt, dnnl::memory::format_tag::abc);
            } else {
                return dnnl::memory::desc({E, N}, dt, dnnl::memory::format_tag::ab);
            }
        };

        auto gk = std::make_shared<grouped_onednn_kernel>();
        gk->has_zp = has_zp;

        gk->gate_pd = make_pd(K_gu, N_gu, _gate_up_group_size, gw_dt);
        gk->gate_prim = dnnl::matmul(gk->gate_pd);
        gk->gate_scale_md = make_quant_md(num_experts, K_gu, _gate_up_group_size, N_gu, dnnl::memory::data_type::f16);
        if (has_zp)
            gk->gate_zp_md = make_quant_md(num_experts, K_gu, _gate_up_group_size, N_gu, gw_dt);

        gk->up_pd = make_pd(K_gu, N_gu, _gate_up_group_size, uw_dt);
        gk->up_prim = dnnl::matmul(gk->up_pd);
        gk->up_scale_md = gk->gate_scale_md;
        if (has_zp)
            gk->up_zp_md = gk->gate_zp_md;

        gk->down_pd = make_pd(K_d, N_d, _down_group_size, dw_dt);
        gk->down_prim = dnnl::matmul(gk->down_pd);
        gk->down_scale_md = make_quant_md(num_experts, K_d, _down_group_size, N_d, dnnl::memory::data_type::f16);
        if (has_zp)
            gk->down_zp_md = make_quant_md(num_experts, K_d, _down_group_size, N_d, dw_dt);

        _grouped_kernels.add(key, gk);
        return *_grouped_kernels.get(key);
    }

    //  inputs 0 is hidden_states, inputs 1 is router_logits[num_tokens, NUM_EXPERTS=128]
    //  extra step Softmax_TopK is fused to give topk-id & router_weights
    //
    //     scratch.topk_id, scratch.full_router_weights = Softmax_TopK(router_logits)
    //
    //  generate expert_mask from topk-id
    //        expert_mask.batch[i][j] : j'th token index for i'th expert
    //        expert_mask.topk[i][j] : topk-output offset for j'th token for i'th expert, used to get weights
    //        expert_mask.pred_flag[i]: bool, if expert i can be skipped
    //
    //     scratch.x, scratch.routing_weights = gather(hidden_states, scratch.full_router_weights, expert_mask.batch, expert_mask.topk)
    //     scratch.y = MLP(scratch.x, .gate/up/down) * scratch.routing_weights
    //     scatter(final_hidden, scratch.y, expert_mask.batch)
    //
    cldnn::event::ptr exec_prefill_onednn(const std::vector<cldnn::event::ptr>& events,
                                          cldnn::stream& stream,
                                          typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                          scratch_buffers& scratch) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& dnn_stream = stream.get_onednn_stream();
        cldnn::event::ptr result_event = nullptr;

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto& engine = instance.get_network().get_engine();
        init_dnnl_weights(cur_moe, engine, scratch.moe_fusion_wei_addr);

        auto routing_mem_ptr = scratch.topk_weights;
        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto get_best_lws = [](size_t hidden_size) {
            const size_t candidate[] = {128, 64, 32, 16, 8};
            for (size_t i = 0; i < sizeof(candidate) / sizeof(size_t); i++) {
                if (hidden_size % candidate[i] == 0) {
                    return candidate[i];
                }
            }
            OPENVINO_THROW("hidden_size=", hidden_size, " is not divisible by any of ", sizeof(candidate) / sizeof(size_t), " candidates");
        };
        auto lws_size = get_best_lws(_hidden_size);
        auto max_topk = static_cast<int64_t>(config.top_k);

        // [batch, max_topk]
        auto topk_id_mem = scratch.topk_id;
        auto token_num = get_seq_len(hidden_states_layout);
        expert_mask_cpu expert_mask;
        get_expert_mask_from_gpu(config, topk_id_mem, stream, expert_mask, token_num);

        for (size_t expert_no = 0; expert_no < config.num_expert; expert_no++) {
            if (expert_no >= expert_mask.pred_flag.size()) {
                OPENVINO_THROW("expert_no=", expert_no, " is out of bounds");
            }
            auto can_skip_subgraph = expert_mask.pred_flag[expert_no] == 0;
            if (can_skip_subgraph) {
                continue;
            }

            on_load_expert_weights(expert_no, instance, dnn_stream);
            auto& dnnl_weights = _dnnl_weights[expert_no];

            // expert_mask
            expert_mask_gpu& expert_mask_mem = scratch.expert_masks[expert_no];
            copy_expert_mask_to_gpu(stream, expert_mask, expert_no, expert_mask_mem);

            auto n_token = static_cast<int>(expert_mask.batch[expert_no].size());

            // Be careful about possible overflow
            if (n_token > std::numeric_limits<int64_t>::max() / max_topk)
                OPENVINO_THROW("n_token * max_topk overflow detected, n_token=", n_token, " max_topk=", max_topk);

            int64_t routing_weights_size = static_cast<int64_t>(n_token * max_topk);
            onednn_kernel& kernel = get_kernel(n_token, static_cast<int>(expert_no), instance);

            // gather
            result_event = execute_stage({result_event},
                                         instance,
                                         *gather,
                                         {hidden_states_mem_ptr, routing_mem_ptr, expert_mask_mem.batch, expert_mask_mem.topk},
                                         {scratch.x, scratch.routing_weights},
                                         {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                                         {1, lws_size},
                                         instance.needs_completion_event());

            // up
            kernel.up.forward(dnn_stream,
                              n_token,
                              convert2dnnl(scratch.x, {static_cast<int64_t>(n_token), dnnl_weights[1].ic}, dnnl::memory::format_tag::ab),
                              convert2dnnl(scratch.up, {static_cast<int64_t>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                              dnnl::memory());
            // gate
            kernel.gate.forward(dnn_stream,
                                n_token,
                                convert2dnnl(scratch.x, {static_cast<int64_t>(n_token), dnnl_weights[0].ic}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.gate, {static_cast<int64_t>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.up, {static_cast<int64_t>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab));
            // down
            kernel.down.forward(dnn_stream,
                                n_token,
                                convert2dnnl(scratch.gate, {static_cast<int64_t>(n_token), _intermediate_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.y, {static_cast<int64_t>(n_token), _hidden_size}, dnnl::memory::format_tag::ab),
                                convert2dnnl(scratch.routing_weights, {static_cast<int64_t>(routing_weights_size)}, dnnl::memory::format_tag::a));
            // index_add
            result_event = execute_stage({result_event},
                                         instance,
                                         *scatter,
                                         {scratch.y, expert_mask_mem.batch},
                                         {final_hidden_states_mem_ptr},
                                         {static_cast<size_t>(n_token), static_cast<size_t>(_hidden_size)},
                                         {1, lws_size},
                                         true /*instance.needs_completion_event()*/);
        }

        return result_event;
    }

    // Returns the weight/ZP memory consumed by the grouped GEMM for one projection:
    // the unpacked u4 scratch for u2 GEMMs, the original buffer otherwise.
    const memory::ptr& grouped_weight_mem(int gemm_idx, const scratch_buffers& scratch) const {
        return _gemm_weights_u2[gemm_idx] ? _u2_unpack_weight[gemm_idx] : scratch.moe_fusion_wei_addr.weight[gemm_idx];
    }
    const memory::ptr& grouped_zp_mem(int gemm_idx, const scratch_buffers& scratch) const {
        return _gemm_weights_u2[gemm_idx] ? _u2_unpack_zp[gemm_idx] : scratch.moe_fusion_wei_addr.zp[gemm_idx];
    }

    // Builds the work-block list for the native u2 GEMM and uploads it.
    //
    // The gathered token buffer is already sorted by expert, but a work-group must never
    // straddle an expert boundary (it would apply one expert's weights to another's tokens).
    // So each active expert contributes ceil(n_e / TILE_M) blocks of {expert_id, token_start,
    // n_tokens}; a short final block is handled by the kernel's n_tokens guard rather than by
    // padding memory. Returns the number of blocks.
    int build_u2_gemm_blocks(typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                             const std::vector<int32_t>& experts_id_cpu,
                             const std::vector<int32_t>& tokens_lens_per_expert_cpu) {
        // 32 rows for the DPAS variant (4 float8 accumulators), 8 for the FMA one. The kernel
        // guards the ragged final block with n_tokens, so a short block costs nothing but a few
        // masked-off accumulator slots.
        const int tile_m = use_u2_dpas() ? MoE3GemmSwigluU2Gemm::DPAS_TILE_M : MoE3GemmSwigluU2Gemm::TILE_M;
        std::vector<int32_t> blocks;
        blocks.reserve(static_cast<size_t>(tokens_lens_per_expert_cpu.size()) * 3);

        int token_start = 0;
        for (size_t e = 0; e < tokens_lens_per_expert_cpu.size(); e++) {
            const int n_e = tokens_lens_per_expert_cpu[e];
            if (n_e <= 0) {
                continue;  // inactive expert contributes no rows and no blocks
            }
            const int expert_id = experts_id_cpu[e];
            for (int off = 0; off < n_e; off += tile_m) {
                blocks.push_back(expert_id);
                blocks.push_back(token_start + off);
                blocks.push_back(std::min(tile_m, n_e - off));
            }
            token_start += n_e;
        }
        if (blocks.empty()) {
            return 0;
        }

        auto& engine = instance.get_network().get_engine();
        const auto bytes = blocks.size() * sizeof(int32_t);
        if (!_u2_gemm_blocks || _u2_gemm_blocks->size() < bytes) {
            auto layout = cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(blocks.size())},
                                        ov::element::i32,
                                        cldnn::format::bfyx);
            // usm_host: written by the CPU every prefill, read once by the kernel. Tiny
            // (3 ints per TILE_M rows), so the host-side write is not worth a staging copy.
            _u2_gemm_blocks = engine.allocate_memory(layout, allocation_type::usm_host, false);
        }
        std::memcpy(_u2_gemm_blocks->buffer_ptr(), blocks.data(), bytes);
        return static_cast<int>(blocks.size() / 3);
    }

    // One native u2 GEMM: C[total, N] = A[total, K] x W[expert, N, K], dequantising u2 on the fly.
    cldnn::event::ptr run_u2_gemm(const cldnn::event::ptr& ev,
                                  typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                  Stage& stage,
                                  int gemm_idx,
                                  const memory::ptr& src,
                                  const memory::ptr& dst,
                                  scratch_buffers& scratch,
                                  size_t n_cols,
                                  int num_blocks) {
        // Fail loudly rather than silently decoding a u4/u8 buffer as u2. The throw is caught by
        // the caller's try/catch, which falls back to the batched GEMV.
        OPENVINO_ASSERT(_gemm_weights_u2[gemm_idx], "moe_3gemm native u2 GEMM invoked on a non-u2 weight, gemm_idx=", gemm_idx);
        const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;
        // The raw u2 weights, NOT the u4 unpack scratch — that is the entire point.
        const auto& wei = scratch.moe_fusion_wei_addr.weight[gemm_idx];
        const auto& scale = scratch.moe_fusion_wei_addr.scale[gemm_idx];
        // INT2_SYM still emits a (scalar) zp; fall back to the weight buffer as a dummy binding
        // when there is none, since the kernel only dereferences it under HAS_ZP.
        const auto& zp = scratch.moe_fusion_wei_addr.zp[gemm_idx] ? scratch.moe_fusion_wei_addr.zp[gemm_idx] : wei;

        if (use_u2_dpas()) {
            // Split N back across work-groups. The FMA variant sweeps all of N inside one
            // work-group because repeating its 33 KB SLM staging 16x was worse; with the staging
            // gone that trade reverses, and narrow N tiles now lose only because each work-group
            // re-reads the activation tile from L2. Measured on the 1k gate/up shape: 32 channels
            // per work-group 3.27 ms, 64 -> 1.43, 128 -> 1.25, 256 -> 1.17, all of N -> 1.15.
            // 256 is the balance point - within 1.5% of the best at 1k, the best measured at 55
            // tokens, and unlike full-N it leaves enough work-groups to load-balance.
            //
            // Dimension order is load-bearing: get_group_id(0) is the N tile so it varies fastest
            // and the work-groups sharing an activation tile stay co-resident in L2, while
            // get_group_id(2) comes out as the block index.
            constexpr size_t sg = MoE3GemmSwigluU2Gemm::DPAS_SG;
            constexpr size_t n_sg = MoE3GemmSwigluU2Gemm::DPAS_N_SG;
            constexpr size_t n_per_wg = MoE3GemmSwigluU2Gemm::DPAS_N_PER_WG;
            OPENVINO_ASSERT(n_cols % n_per_wg == 0,
                            "moe_3gemm u2 DPAS GEMM N=",
                            n_cols,
                            " is not a multiple of ",
                            n_per_wg);
            return execute_stage({ev},
                                 instance,
                                 stage,
                                 {src, wei, scale, zp, _u2_gemm_blocks},
                                 {dst},
                                 {n_cols / n_per_wg, sg, n_sg * static_cast<size_t>(num_blocks)},
                                 {1, sg, n_sg});
        }

        OPENVINO_ASSERT(n_cols % (N_BLOCK * SUBGROUP_NUM) == 0,
                        "moe_3gemm u2 GEMM N=",
                        n_cols,
                        " is not a multiple of N_BLOCK*SUBGROUP_NUM=",
                        N_BLOCK * SUBGROUP_NUM);
        // Exactly one work-group along N: the kernel sweeps the whole N range internally so the
        // per-token SLM staging is done once per token block instead of once per N block.
        return execute_stage({ev},
                             instance,
                             stage,
                             {src, wei, scale, zp, _u2_gemm_blocks},
                             {dst},
                             {static_cast<size_t>(num_blocks), subgroup_size, SUBGROUP_NUM},
                             {1, subgroup_size, SUBGROUP_NUM});
    }

    // u2 prefill: unpack the u2 expert weights (and ZP) into u4-packed scratch buffers so
    // the grouped GEMM path (which has no u2 dtype) can serve prefill. The scratch buffers are
    // allocated on first use and the unpack runs ONCE per bound weight buffer, not per prefill:
    // the weights are constant (OTD is rejected for u2 in the ctor), so re-unpacking identical
    // data every prefill was pure overhead — a full streaming pass over every expert of every
    // layer, which measured ~430ms of TTFT per prefill on Qwen3.6-35B-A3B and made a multi-turn
    // benchmark ~1.6x slower end-to-end than the same model at int4. See the *_key members.
    //
    // skip_routed: the native u2 GEMM consumes the routed-expert weights directly, so their u4
    // scratch (12.08 GB on Qwen3.6-35B-A3B — the entire reason this function was expensive) is
    // never allocated. The shared expert still goes through oneDNN primitives that have no u2
    // dtype, so its four much smaller buffers are still unpacked.
    cldnn::event::ptr unpack_u2_weights_for_prefill(typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                                    scratch_buffers& scratch,
                                                    const std::vector<cldnn::event::ptr>& events,
                                                    bool skip_routed = false) {
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& engine = instance.get_network().get_engine();
        cldnn::event::ptr ret = events.empty() ? nullptr : events[0];

        const size_t lws = 256;
        const auto round_up = [lws](size_t v) {
            return (v + lws - 1) / lws * lws;
        };
        for (int i = 0; i < 3 && !skip_routed; i++) {
            if (!_gemm_weights_u2[i])
                continue;

            // Weights: byte-wise u2 -> u4 unpack, doubles the buffer size. The u2 and u4
            // expert weight buffers share the same logical element order ([expert, oc, ic],
            // ic innermost, LSB-first packing), so a straight byte-wise unpack reproduces
            // exactly the layout the u4 prefill paths expect.
            const auto& src_mem = scratch.moe_fusion_wei_addr.weight[i];
            const size_t src_bytes = src_mem->size();
            OPENVINO_ASSERT(src_bytes % 4 == 0, "moe_3gemm u2 weight buffer byte size ", src_bytes, " is not a multiple of 4");
            const bool wei_realloc = !_u2_unpack_weight[i] || _u2_unpack_weight[i]->size() < src_bytes * 2;
            if (wei_realloc) {
                auto dst_layout =
                    cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(src_bytes * 2)}, ov::element::u8, cldnn::format::bfyx);
                _u2_unpack_weight[i] = engine.allocate_memory(dst_layout, allocation_type::usm_device, false);
            }
            // A fresh allocation is uninitialised, so it always forces the unpack to run.
            if (wei_realloc || !_u2_unpack_weight_key[i].matches(src_mem)) {
                const int src_uints = static_cast<int>(src_bytes / 4);
                ret = execute_stage({ret}, instance, *u2_unpack, {src_mem}, {_u2_unpack_weight[i]}, {round_up(src_bytes / 4), 1, 1}, {lws, 1, 1},
                                    false,
                                    {src_uints, 0});
                _u2_unpack_weight_key[i].set(src_mem);
            }

            // ZP: u2-packed ZP is unpacked like the weights; a scalar (per-tensor) ZP is
            // broadcast into a full u4 ZP tensor [num_expert, num_groups, oc].
            if (!config.has_zp)
                continue;
            const auto& zp_src = scratch.moe_fusion_wei_addr.zp[i];
            OPENVINO_ASSERT(zp_src, "moe_3gemm u2 GEMM ", i, " has_zp but no ZP buffer bound");
            size_t zp_dst_bytes = 0;
            size_t work_items = 0;
            int zp_mode = 0;
            int zp_count = 0;
            if (zp_src->get_layout().count() == 1) {
                // Size matches make_quant_md() in get_grouped_kernel: [num_expert, K / group_size, oc] u4.
                const size_t k = (i == 2) ? static_cast<size_t>(_intermediate_size) : static_cast<size_t>(_hidden_size);
                const size_t oc = (i == 2) ? static_cast<size_t>(_hidden_size) : static_cast<size_t>(_intermediate_size);
                const size_t gs = (i == 2) ? static_cast<size_t>(_down_group_size) : static_cast<size_t>(_gate_up_group_size);
                const size_t num_groups = (gs == 0 || gs >= k) ? 1 : (k / gs);
                zp_dst_bytes = static_cast<size_t>(config.num_expert) * num_groups * oc / 2;
                work_items = zp_dst_bytes;  // one output byte per work item
                zp_mode = 1;
                zp_count = static_cast<int>(zp_dst_bytes);
            } else if (zp_src->get_layout().data_type == data_types::u2) {
                // u2-packed zp: same byte-wise unpack as the weights.
                const size_t zp_src_bytes = zp_src->size();
                OPENVINO_ASSERT(zp_src_bytes % 4 == 0, "moe_3gemm u2 zp buffer byte size ", zp_src_bytes, " is not a multiple of 4");
                zp_dst_bytes = zp_src_bytes * 2;
                work_items = zp_src_bytes / 4;  // one input uint per work item
                zp_mode = 0;
                zp_count = static_cast<int>(zp_src_bytes / 4);
            } else {
                // Byte-wide zp (u8/i8, e.g. per-channel or per-group): one VALUE per
                // byte, NOT u2-packed data. Pack two adjacent values into one u4
                // byte (mode 2); a byte-wise u2 unpack would double the values into
                // garbage and silently corrupt prefill dequant.
                const size_t zp_vals = zp_src->get_layout().count();
                const size_t k = (i == 2) ? static_cast<size_t>(_intermediate_size) : static_cast<size_t>(_hidden_size);
                const size_t oc = (i == 2) ? static_cast<size_t>(_hidden_size) : static_cast<size_t>(_intermediate_size);
                const size_t gs = (i == 2) ? static_cast<size_t>(_down_group_size) : static_cast<size_t>(_gate_up_group_size);
                const size_t num_groups = (gs == 0 || gs >= k) ? 1 : (k / gs);
                const size_t expected = static_cast<size_t>(config.num_expert) * num_groups * oc;
                OPENVINO_ASSERT(zp_vals == expected,
                                "moe_3gemm u2 zp element count ",
                                zp_vals,
                                " does not match grouped GEMM zp descriptor [",
                                config.num_expert,
                                ", ",
                                num_groups,
                                ", ",
                                oc,
                                "]");
                zp_dst_bytes = (zp_vals + 1) / 2;
                work_items = zp_dst_bytes;  // one output byte per work item
                zp_mode = 2;
                zp_count = static_cast<int>(zp_vals);
            }
            const bool zp_realloc = !_u2_unpack_zp[i] || _u2_unpack_zp[i]->size() < zp_dst_bytes;
            if (zp_realloc) {
                auto dst_layout =
                    cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(zp_dst_bytes)}, ov::element::u8, cldnn::format::bfyx);
                _u2_unpack_zp[i] = engine.allocate_memory(dst_layout, allocation_type::usm_device, false);
            }
            if (zp_realloc || !_u2_unpack_zp_key[i].matches(zp_src)) {
                ret = execute_stage({ret}, instance, *u2_unpack, {zp_src}, {_u2_unpack_zp[i]}, {round_up(work_items), 1, 1}, {lws, 1, 1}, false,
                                    {zp_count, zp_mode});
                _u2_unpack_zp_key[i].set(zp_src);
            }
        }

        // Shared expert: same u2 -> u4 unpack per projection (gate/up/down/scalar-gate),
        // consumed by the oneDNN shared-expert primitives built in init_shared_primitives.
        // The u2 and u4 buffers share the logical element order ([oc, ic], ic innermost),
        // so a straight byte-wise unpack matches the {ic, oc}/ba u4 descriptors.
        for (int i = 0; i < 4; i++) {
            if (!_shared_weights_u2[i])
                continue;

            const auto& src_mem = scratch.moe_fusion_wei_addr.shared_weight[i];
            OPENVINO_ASSERT(src_mem, "moe_3gemm shared weight ", i, " is u2 but its buffer is not bound");
            const size_t src_bytes = src_mem->size();
            OPENVINO_ASSERT(src_bytes % 4 == 0, "moe_3gemm u2 shared weight buffer byte size ", src_bytes, " is not a multiple of 4");
            const bool wei_realloc = !_u2_unpack_shared_weight[i] || _u2_unpack_shared_weight[i]->size() < src_bytes * 2;
            if (wei_realloc) {
                auto dst_layout =
                    cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(src_bytes * 2)}, ov::element::u8, cldnn::format::bfyx);
                _u2_unpack_shared_weight[i] = engine.allocate_memory(dst_layout, allocation_type::usm_device, false);
            }
            if (wei_realloc || !_u2_unpack_shared_weight_key[i].matches(src_mem)) {
                const int src_uints = static_cast<int>(src_bytes / 4);
                ret = execute_stage({ret}, instance, *u2_unpack, {src_mem}, {_u2_unpack_shared_weight[i]}, {round_up(src_bytes / 4), 1, 1},
                                    {lws, 1, 1},
                                    false,
                                    {src_uints, 0});
                _u2_unpack_shared_weight_key[i].set(src_mem);
            }

            // The scalar gate (i == 3) has no scale/zp (onednn_linear with ic_group_size = -1).
            if (i == 3)
                continue;
            const auto& zp_src = scratch.moe_fusion_wei_addr.shared_zp[i];
            if (!zp_src || zp_src->get_layout().data_type == data_types::dynamic)
                continue;  // symmetric placeholder: no zp to unpack
            // ZP sizes match the descriptors built in init_shared_primitives:
            // gate/up: {_hidden_size / _gate_up_group_size, _shared_intermediate_size} u4,
            // down: {_shared_intermediate_size / _down_group_size, _hidden_size} u4.
            size_t zp_dst_bytes = 0;
            size_t work_items = 0;
            int zp_mode = 0;
            int zp_count = 0;
            if (zp_src->get_layout().count() == 1) {
                // Scalar (per-tensor) zp: broadcast into the full u4 zp tensor.
                const size_t k = (i == 2) ? static_cast<size_t>(_shared_intermediate_size) : static_cast<size_t>(_hidden_size);
                const size_t oc = (i == 2) ? static_cast<size_t>(_hidden_size) : static_cast<size_t>(_shared_intermediate_size);
                const size_t gs = (i == 2) ? static_cast<size_t>(_down_group_size) : static_cast<size_t>(_gate_up_group_size);
                const size_t num_groups = (gs == 0 || gs >= k) ? 1 : (k / gs);
                zp_dst_bytes = num_groups * oc / 2;
                work_items = zp_dst_bytes;  // one output byte per work item
                zp_mode = 1;
                zp_count = static_cast<int>(zp_dst_bytes);
            } else if (zp_src->get_layout().data_type == data_types::u2) {
                const size_t zp_src_bytes = zp_src->size();
                OPENVINO_ASSERT(zp_src_bytes % 4 == 0, "moe_3gemm u2 shared zp buffer byte size ", zp_src_bytes, " is not a multiple of 4");
                zp_dst_bytes = zp_src_bytes * 2;
                work_items = zp_src_bytes / 4;  // one input uint per work item
                zp_mode = 0;
                zp_count = static_cast<int>(zp_src_bytes / 4);
            } else {
                // Byte-wide zp (u8/i8): pack two values per u4 byte (mode 2), same as
                // the routed-expert path; a byte-wise u2 unpack would corrupt it.
                const size_t zp_vals = zp_src->get_layout().count();
                const size_t k = (i == 2) ? static_cast<size_t>(_shared_intermediate_size) : static_cast<size_t>(_hidden_size);
                const size_t oc = (i == 2) ? static_cast<size_t>(_hidden_size) : static_cast<size_t>(_shared_intermediate_size);
                const size_t gs = (i == 2) ? static_cast<size_t>(_down_group_size) : static_cast<size_t>(_gate_up_group_size);
                const size_t num_groups = (gs == 0 || gs >= k) ? 1 : (k / gs);
                OPENVINO_ASSERT(zp_vals == num_groups * oc,
                                "moe_3gemm u2 shared zp element count ",
                                zp_vals,
                                " does not match shared-expert zp descriptor [",
                                num_groups,
                                ", ",
                                oc,
                                "]");
                zp_dst_bytes = (zp_vals + 1) / 2;
                work_items = zp_dst_bytes;
                zp_mode = 2;
                zp_count = static_cast<int>(zp_vals);
            }
            const bool zp_realloc = !_u2_unpack_shared_zp[i] || _u2_unpack_shared_zp[i]->size() < zp_dst_bytes;
            if (zp_realloc) {
                auto dst_layout =
                    cldnn::layout({1, 1, 1, static_cast<ov::Dimension::value_type>(zp_dst_bytes)}, ov::element::u8, cldnn::format::bfyx);
                _u2_unpack_shared_zp[i] = engine.allocate_memory(dst_layout, allocation_type::usm_device, false);
            }
            if (zp_realloc || !_u2_unpack_shared_zp_key[i].matches(zp_src)) {
                ret = execute_stage({ret}, instance, *u2_unpack, {zp_src}, {_u2_unpack_shared_zp[i]}, {round_up(work_items), 1, 1}, {lws, 1, 1},
                                    false,
                                    {zp_count, zp_mode});
                _u2_unpack_shared_zp_key[i].set(zp_src);
            }
        }
        return ret;
    }

    // Third prefill path: OneDNN grouped GEMM (one matmul call per GEMM layer, all experts together).
    // This avoids the per-expert loop of exec_prefill_onednn while keeping full weight-format
    // compatibility (quantized or fp16 weights).
    //
    //  gather_by_expert(hidden_states, topk_id) -> scratch.x          [total, hidden]
    //  grouped_matmul(scratch.x, W_gate)        -> scratch.gate       [total, inter]
    //  grouped_matmul(scratch.x, W_up)          -> scratch.up         [total, inter]
    //  silu(scratch.gate) * scratch.up          -> scratch.gate       [total, inter]
    //  grouped_matmul(scratch.gate, W_down)     -> scratch.y          [total, hidden]
    //  scatter_reduce(scratch.y, topk_id, topk_weights) -> output     [token_num, hidden]
    //
    // Note: "total" = token_num * max_topk, sorted by expert assignment.
    //
    cldnn::event::ptr exec_prefill_grouped_gemm(const std::vector<cldnn::event::ptr>& events,
                                                cldnn::stream& stream,
                                                typed_primitive_inst<moe_3gemm_fused_compressed>& instance,
                                                scratch_buffers& scratch) {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("moe_3gemm_swiglu_opt_impl::exec_prefill_grouped_gemm"));

        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& dnn_stream = stream.get_onednn_stream();

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        auto token_num = get_seq_len(hidden_states_layout);
        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        auto batch_mem_ptr = scratch.topk_id;
        auto routing_mem_ptr = scratch.topk_weights;
        const auto& intermediates_memories = instance.get_intermediates_memories();

        int num_total_experts = static_cast<int>(config.num_expert);
        int max_topk = static_cast<int>(config.top_k);
        int num_actually_used_experts = 0;

        // In OTD mode, the grouped descriptor dimension is resident_slot_count() (LRU slots)
        // rather than the full num_total_experts.
        int num_grouped_experts = get_num_grouped_experts(num_total_experts);

        // ----------------------------------------------------------------
        // Step 1: CPU mask generation (topk_id already flushed by caller)
        // ----------------------------------------------------------------
        cldnn::event::ptr ret_event = events.empty() ? nullptr : events[0];
        // Flat list of source token indices per expert – input for prefill_gather
        std::vector<int32_t> tokens_per_expert_cpu(static_cast<size_t>(token_num) * max_topk, -1);
        // Compact per-activated-expert metadata reused by scatter_reduce
        std::vector<int32_t> tokens_lens_per_expert_cpu(num_grouped_experts, 0);
        std::vector<int32_t> experts_id_cpu(num_grouped_experts, -1);
        // int32_t cumulative end-offsets per expert/slot for OneDNN grouped GEMM
        // offsets[e] = sum(n_0..n_e) = exclusive end of expert/slot e in the flat buffer.
        // This is the s32 format expected by dnnl::memory::desc::grouped().
        std::vector<int32_t> grouped_offsets_cpu(num_grouped_experts, 0);

        if (!build_grouped_mask_otd(stream,
                                    instance,
                                    scratch,
                                    batch_mem_ptr,
                                    token_num,
                                    max_topk,
                                    num_grouped_experts,
                                    tokens_per_expert_cpu,
                                    tokens_lens_per_expert_cpu,
                                    experts_id_cpu,
                                    grouped_offsets_cpu,
                                    num_actually_used_experts,
                                    events)) {
            if (_weight_provider->is_offloaded()) {
                // OTD: unique experts > capacity → explicit fallback to per-expert loop
                return exec_prefill_onednn(events, stream, instance, scratch);
            }
            // Non-OTD path: build mask from original expert IDs
            expert_mask_cpu expert_mask;
            get_expert_mask_from_gpu(config, batch_mem_ptr, stream, expert_mask, token_num);

            int tokens_iter = 0;
            int experts_iter = 0;
            int32_t running_offset = 0;
            for (int e = 0; e < num_total_experts; e++) {
                auto n = static_cast<int32_t>(expert_mask.batch[e].size());
                running_offset += n;
                grouped_offsets_cpu[e] = running_offset;  // exclusive end of expert e
                if (n > 0) {
                    experts_id_cpu[experts_iter] = e;
                    tokens_lens_per_expert_cpu[experts_iter] = n;
                    ++experts_iter;
                    ++num_actually_used_experts;
                    for (auto t : expert_mask.batch[e])
                        tokens_per_expert_cpu[tokens_iter++] = t;
                }
            }
        }

        int total_gathered_tokens = static_cast<int>(token_num) * max_topk;

        // Compute actual max tokens assigned to any single expert.
        int max_tokens_per_expert = 0;
        if (num_actually_used_experts > 0) {
            max_tokens_per_expert = *std::max_element(tokens_lens_per_expert_cpu.begin(), tokens_lens_per_expert_cpu.begin() + num_actually_used_experts);
        }

        GPU_DEBUG_TRACE_DETAIL << "\nexec_prefill_grouped_gemm: token_num=" << token_num << ", total_gathered_tokens=" << total_gathered_tokens
                               << ", max_tokens_per_expert=" << max_tokens_per_expert << ", num_actually_used_experts=" << num_actually_used_experts
                               << std::endl;

        // Upload scratch metadata for the scatter_reduce and gather kernels
        intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]
            ->copy_from(stream, tokens_per_expert_cpu.data(), 0, 0, tokens_per_expert_cpu.size() * sizeof(int32_t), true);
        // When ONEDNN_GROUPED_GEMM_USED, the scatter_reduce kernel reads:
        //   exp_offset_start = expert_id == 0 ? 0 : experts_start_offset[expert_id - 1]
        // So experts_start_offset[k] must equal the start index of expert k+1 in the flat buffer
        // = exclusive end of expert k = grouped_offsets_cpu[k].
        {
            std::vector<int32_t> expert_start_offsets_per_id(static_cast<size_t>(num_grouped_experts - 1));
            for (int e = 0; e < num_grouped_experts - 1; ++e)
                expert_start_offsets_per_id[e] = grouped_offsets_cpu[e];  // end[e] == start[e+1]
            intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT]
                ->copy_from(stream, expert_start_offsets_per_id.data(), 0, 0, expert_start_offsets_per_id.size() * sizeof(int32_t), true);
        }
        intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS]
            ->copy_from(stream, experts_id_cpu.data(), 0, 0, static_cast<size_t>(num_actually_used_experts) * sizeof(int32_t), true);
        intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT]
            ->copy_from(stream, tokens_lens_per_expert_cpu.data(), 0, 0, static_cast<size_t>(num_actually_used_experts) * sizeof(int32_t), true);
        intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]->copy_from(stream, &num_actually_used_experts, 0, 0, sizeof(int32_t), true);
        // int32_t end-offsets for the OneDNN grouped descriptor
        intermediates_memories[MOE_INTERNAL_BUFFER_GROUPED_OFFSETS]
            ->copy_from(stream, grouped_offsets_cpu.data(), 0, 0, grouped_offsets_cpu.size() * sizeof(int32_t), true);

        // ----------------------------------------------------------------
        // Step 2: GPU gather – reorder input tokens sorted by expert
        // ----------------------------------------------------------------
        {
            auto hidden_size = _hidden_size;
            auto block_size = get_vec_size(*instance.get_impl_params());
            auto [local_threads_count, batches_per_thread, unaligned_elements] =
                calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), block_size, hidden_size);

            ret_event = execute_stage({ret_event},
                                      instance,
                                      *grouped_gemm_prefill_gather,
                                      {instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)),
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT]},
                                      {scratch.x},
                                      {static_cast<size_t>(total_gathered_tokens) * local_threads_count, 1, 1},
                                      {local_threads_count, 1, 1});
        }

        // In OTD mode, ensure gather OCL kernel completes before OneDNN reads scratch.x.
        // Non-OTD relies on the in-order queue's implicit ordering.
        on_before_grouped_gather(stream);

        // ----------------------------------------------------------------
        // Steps 3-5: OneDNN grouped GEMM – gate, up, SiLU, down
        // ----------------------------------------------------------------
        // SAFETY CHECK: Verify grouped_offsets buffer exists before accessing
        if (intermediates_memories.size() <= MOE_INTERNAL_BUFFER_GROUPED_OFFSETS) {
            OPENVINO_THROW("[MOE_3GEMM_GROUPED_BUG] Grouped GEMM path requires buffer ",
                           MOE_INTERNAL_BUFFER_GROUPED_OFFSETS,
                           " (GROUPED_OFFSETS) but only ",
                           intermediates_memories.size(),
                           " buffers allocated. ",
                           "This indicates a mismatch between buffer allocation and execution path. ",
                           "use_grouped_gemm_prefill=",
                           use_grouped_gemm_prefill);
        }
        // Native u2 path: the three expert GEMMs run on the u2 weights directly, so no oneDNN
        // grouped primitive is built and no u4 unpack scratch is allocated. Everything around
        // them (gather, SiLU*mul, scatter_reduce, the shared expert) is unchanged.
        const bool use_native_u2 = use_native_u2_prefill();
        int u2_num_blocks = 0;
        if (use_native_u2) {
            u2_num_blocks = build_u2_gemm_blocks(instance, experts_id_cpu, tokens_lens_per_expert_cpu);
            OPENVINO_ASSERT(u2_num_blocks > 0, "moe_3gemm native u2 GEMM: no work blocks for ", total_gathered_tokens, " gathered tokens");
        }

        // get_grouped_kernel() builds oneDNN primitives against u4 weight descriptors; on the
        // native path those weights do not exist, so it must not be called at all.
        grouped_onednn_kernel* gk_ptr = use_native_u2 ? nullptr : &get_grouped_kernel(total_gathered_tokens, instance);
        auto row_offsets = intermediates_memories[MOE_INTERNAL_BUFFER_GROUPED_OFFSETS];

        // Runtime dispatch hint: actual max tokens assigned to any single expert.
        // Passed as DNNL_ARG_HINT_MAX_GROUP_SIZE to each grouped matmul execute(),
        // allowing the kernel to reduce per-expert workgroup dispatch without
        // recompiling the primitive.
        auto hint_md = dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32);
        dnnl::memory hint_mem(hint_md, static_cast<int32_t>(max_tokens_per_expert));

        // gate GEMM: [total, hidden] * W_gate[E,hidden,inter] -> [total, inter]
        if (use_native_u2) {
            ret_event = run_u2_gemm(ret_event, instance, *u2_gemm_gate, 0, scratch.x, scratch.gate, scratch,
                                    static_cast<size_t>(_intermediate_size), u2_num_blocks);
        } else {
            auto& gk = *gk_ptr;
            auto src_mem = scratch.x->get_onednn_grouped_memory(gk.gate_pd.src_desc(), *row_offsets);
            auto dst_mem = scratch.gate->get_onednn_grouped_memory(gk.gate_pd.dst_desc(), *row_offsets);
            auto w_mem = grouped_weight_mem(0, scratch)->get_onednn_memory(gk.gate_pd.weights_desc());
            auto scale_mem = scratch.moe_fusion_wei_addr.scale[0]->get_onednn_memory(gk.gate_scale_md);

            std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, src_mem},
                                                       {DNNL_ARG_WEIGHTS, w_mem},
                                                       {DNNL_ARG_DST, dst_mem},
                                                       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
                                                       {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}};
            if (gk.has_zp) {
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, grouped_zp_mem(0, scratch)->get_onednn_memory(gk.gate_zp_md)});
            }
            gk.gate_prim.execute(dnn_stream, args);
        }

        // up GEMM: [total, hidden] * W_up[E,hidden,inter] -> [total, inter]
        if (use_native_u2) {
            ret_event = run_u2_gemm(ret_event, instance, *u2_gemm_up, 1, scratch.x, scratch.up, scratch,
                                    static_cast<size_t>(_intermediate_size), u2_num_blocks);
        } else {
            auto& gk = *gk_ptr;
            auto src_mem = scratch.x->get_onednn_grouped_memory(gk.up_pd.src_desc(), *row_offsets);
            auto dst_mem = scratch.up->get_onednn_grouped_memory(gk.up_pd.dst_desc(), *row_offsets);
            auto w_mem = grouped_weight_mem(1, scratch)->get_onednn_memory(gk.up_pd.weights_desc());
            auto scale_mem = scratch.moe_fusion_wei_addr.scale[1]->get_onednn_memory(gk.up_scale_md);

            std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, src_mem},
                                                       {DNNL_ARG_WEIGHTS, w_mem},
                                                       {DNNL_ARG_DST, dst_mem},
                                                       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
                                                       {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}};
            if (gk.has_zp) {
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, grouped_zp_mem(1, scratch)->get_onednn_memory(gk.up_zp_md)});
            }
            gk.up_prim.execute(dnn_stream, args);
        }

        // Step 4: SiLU(gate) * up -> scratch.gate  (OCL prefill_swiglu kernel, compiled with ONEDNN_GROUPED_GEMM_USED)
        // gate and up GEMMs are submitted to the same OCL queue as the OCL kernels;
        // passing ret_event (from gather) as dependency ensures ordering within the queue.
        {
            const size_t subgroup_size = instance.get_impl_params()->get_device_info().arch >= gpu_arch::xe2 ? 32 : 16;

            ret_event = execute_stage({ret_event},
                                      instance,
                                      *grouped_gemm_prefill_swiglu,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_UP_OUTPUT], intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_GATE_OUTPUT]},
                                      {static_cast<size_t>(_intermediate_size), static_cast<size_t>(total_gathered_tokens), 1},
                                      {subgroup_size, 1, 1});
        }

        // down GEMM: [total, inter] * W_down[E,inter,hidden] -> [total, hidden]
        if (use_native_u2) {
            ret_event = run_u2_gemm(ret_event, instance, *u2_gemm_down, 2, scratch.gate, scratch.y, scratch,
                                    static_cast<size_t>(_hidden_size), u2_num_blocks);
        } else {
            auto& gk = *gk_ptr;
            auto src_mem = scratch.gate->get_onednn_grouped_memory(gk.down_pd.src_desc(), *row_offsets);
            auto dst_mem = scratch.y->get_onednn_grouped_memory(gk.down_pd.dst_desc(), *row_offsets);
            auto w_mem = grouped_weight_mem(2, scratch)->get_onednn_memory(gk.down_pd.weights_desc());
            auto scale_mem = scratch.moe_fusion_wei_addr.scale[2]->get_onednn_memory(gk.down_scale_md);

            std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_SRC, src_mem},
                                                       {DNNL_ARG_WEIGHTS, w_mem},
                                                       {DNNL_ARG_DST, dst_mem},
                                                       {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
                                                       {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}};
            if (gk.has_zp) {
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, grouped_zp_mem(2, scratch)->get_onednn_memory(gk.down_zp_md)});
            }
            gk.down_prim.execute(dnn_stream, args);
        }

        // Ensure all grouped GEMMs complete before scatter_reduce (OTD sync)
        on_after_grouped_gemm(stream);

        // ----------------------------------------------------------------
        // Step 6: scatter_reduce – weighted accumulate into output
        // ----------------------------------------------------------------
        {
            auto [local_threads_count, batches_per_thread, _unused] =
                calc_thread_count(const_cast<RuntimeParams&>(*instance.get_impl_params()), 4, _hidden_size);

            ret_event = execute_stage({ret_event},
                                      instance,
                                      *grouped_gemm_prefill_scatter_reduce,
                                      {intermediates_memories[MOE_INTERNAL_BUFFER_DOWN_OUTPUT],
                                       batch_mem_ptr,
                                       routing_mem_ptr,
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_IDX_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_START_OFFSET_PER_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_TOKEN_LEN_PER_ACTIVATED_EXPERT],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTIVATED_EXPERT_IDS],
                                       intermediates_memories[MOE_INTERNAL_BUFFER_ACTUAL_USED_EXPERT_NUM]},
                                      {final_hidden_states_mem_ptr},
                                      {static_cast<size_t>(token_num) * local_threads_count, 1, 1},
                                      {local_threads_count, 1, 1},
                                      true /*needs_completion_event*/);
        }

        return ret_event;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, openvino::itt::handle("moe_3gemm_swiglu_opt_impl::execute"));
        auto& instance = reinterpret_cast<typed_primitive_inst<moe_3gemm_fused_compressed>&>(ins);
        auto cur_moe = instance.get_typed_desc<moe_3gemm_fused_compressed>();
        const auto& config = cur_moe->_config;
        auto& cur_net = instance.get_network();
        auto& stream = cur_net.get_stream();

        OPENVINO_ASSERT(_weight_provider, "expert weight provider not initialized");

        cldnn::event::ptr ret_env = nullptr;
        _has_shared_expert = (config.num_shared_expert > 0);

        if (_has_shared_expert) {
            if (config.num_shared_expert > 1) {
                OPENVINO_THROW("num_shared_expert=", config.num_shared_expert, " is not supported yet, only support 0 or 1");
            }
            auto shared_expert_weight_layout = instance.input_memory_ptr(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_WEIGHT))->get_layout();
            auto hidden_size = static_cast<int>(cur_moe->_config.hidden_size);
            _shared_intermediate_size = static_cast<int>(shared_expert_weight_layout.count() / hidden_size);
            OPENVINO_ASSERT(_shared_intermediate_size == _intermediate_size, "Shared expert _intermediate_size should be same with moe experts");
        }

        auto [hidden_states_mem_ptr, hidden_states_layout] = get_input_info(instance, static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        size_t token_num = get_seq_len(hidden_states_layout);

        bind_weights_on_first_exec(instance);

        scratch_buffers scratch;
        prepare_internal_buffers(instance, scratch, token_num);
        kernel_dump_info.clear_entries();

        // Batched GEMV: for small token counts (including single token, MTP/speculative decoding),
        // use optimized GEMV kernels with batch dimension. Avoids gather/scatter overhead.
        // u2 weights: decode takes this path; prefill (token_num > threshold) unpacks the
        // u2 weights (routed and shared expert) into u4 scratch buffers and runs the grouped
        // GEMM path below.
        if (token_num <= batched_gemv_threshold) {
            return exec_batched_gemv(events, instance, scratch, token_num);
        }

        auto final_hidden_states_mem_ptr = instance.output_memory_ptr(0);
        // Pre-zero output buffer for paths that accumulate into it.
        // The per-expert onednn loop uses index_add (accumulates into output).
        // The grouped_gemm scatter_reduce writes all token positions atomically
        // and does not require pre-zeroing — except in OTD mode where the
        // fallback to exec_prefill_onednn (when unique_experts > lru slots)
        // also accumulates via index_add.
        if (!use_micro_gemm_prefill && should_pre_zero_output()) {
            final_hidden_states_mem_ptr->fill(stream, 0u);
        }
        // GPU mask gen is only supported for micro_gemm; both grouped_gemm and onednn loop
        // always use CPU mask gen and therefore always need topk to be ready first.
        const bool use_gpu_mask_gen = use_micro_gemm_prefill && use_gpu_mask_gen_prefill;
        if (!use_gpu_mask_gen) {
            // Wait for input events (topk produced upstream by MoERouterFused)
            for (auto& ev : events) {
                if (ev) {
                    ev->wait();
                }
            }
        }

        GPU_DEBUG_TRACE_DETAIL << "\nMoE3GemmFusedCompressed exec(): token_num=" << token_num << ", max_topk=" << static_cast<int>(config.top_k)
                               << ", use_micro_gemm_prefill=" << use_micro_gemm_prefill << ", use_grouped_gemm_prefill=" << use_grouped_gemm_prefill
                               << std::endl;
        update_rt_params(instance);
        // u2 prefill: unpack the u2 weights (and zp) into u4 scratch buffers. Routed-expert
        // u2 GEMMs are consumed by the grouped GEMM path below; shared-expert u2 projections
        // are consumed by the oneDNN shared-expert primitives (init_shared_primitives), which
        // run on every prefill path, so the unpack is not tied to _weights_u2 alone.
        cldnn::event::ptr unpack_event = nullptr;
        if (_weights_u2 || has_u2_shared_weights()) {
            try {
                // _weights_u2 => the routed experts take the native u2 GEMM below, so only the
                // shared expert's projections still need a u4 copy.
                unpack_event = unpack_u2_weights_for_prefill(instance, scratch, events, /*skip_routed=*/use_native_u2_prefill());
            } catch (const std::exception& e) {
                // The batched GEMV fuses the shared expert in-kernel, so returning it
                // directly also skips the oneDNN shared-expert block below.
                GPU_DEBUG_TRACE_DETAIL << "u2 weight unpack for prefill failed (" << e.what() << "), falling back to batched GEMV" << std::endl;
                instance.output_memory_ptr(0)->fill(stream, false);
                return exec_batched_gemv(events, instance, scratch, token_num);
            }
        }
        if (_weights_u2) {
            // Run the grouped GEMM path on the unpacked u4 weights as regular u4 weights. If
            // the grouped GEMM cannot be created/executed, fall back to the batched GEMV
            // (the pre-u2-prefill behavior).
            try {
                ret_env = exec_prefill_grouped_gemm({unpack_event}, stream, instance, scratch);
            } catch (const std::exception& e) {
                GPU_DEBUG_TRACE_DETAIL << "u2 grouped GEMM prefill failed (" << e.what() << "), falling back to batched GEMV" << std::endl;
                instance.output_memory_ptr(0)->fill(stream, false);
                // Return directly: the batched GEMV fuses the shared expert in-kernel, so it
                // must skip the oneDNN shared-expert block below (it would apply it twice).
                return exec_batched_gemv(events, instance, scratch, token_num);
            }
        } else if (use_micro_gemm_prefill) {
            ret_env = exec_prefill_micro_gemm(events, instance, scratch, use_gpu_mask_gen);
        } else if (use_grouped_gemm_prefill) {
            ret_env = exec_prefill_grouped_gemm(events, stream, instance, scratch);
            // In OTD mode the grouped_gemm path interleaves OCL kernels with OneDNN
            // grouped matmul on the same in-order queue. The framework's event-based
            // scheduling may proceed to subsequent graph nodes before scatter_reduce
            // completes, causing multi-iteration inference to degrade.
            // Flush the queue only in OTD mode to avoid impacting non-OTD perf.
            on_after_exec_sync(stream);
        } else {
            ret_env = exec_prefill_onednn(events, stream, instance, scratch);
        }

        if (_has_shared_expert) {
            auto& engine = instance.get_network().get_engine();
            init_shared_primitives(engine, scratch.moe_fusion_wei_addr, static_cast<int>(token_num));
            // Shared expert's down_proj uses sum post-op (output += result), so the
            // scatter_reduce must have written the MoE output first.  Both are on the
            // same in-order OCL queue, so submission order guarantees execution order.
            // No explicit wait() is needed — the in-order queue serializes all GPU work,
            // and any subsequent primitive on the same queue will see the completed output.
            if (use_grouped_gemm_prefill && ret_env) {
                // ensure grouped GEMM fully completes before executing shared expert, which relies on its output being ready;
                // For grouped_gemm path, scatter_reduce (OCL) is preceded by multiple OCL <--> OneDNN
                // transitions inside exec_prefill_grouped_gemm. The implicit ordering between
                // the OCL queue and OneDNN's stream cannot be relied upon across this many
                // back-and-forth submissions, so the shared expert's down_proj sum post-op
                // (which reads+writes final_hidden_states) can race with scatter_reduce.
                // Force the scatter_reduce write to be visible before submitting the shared expert.
                ret_env->wait();
            }
            execute_shared_expert(stream.get_onednn_stream(), static_cast<int>(token_num), hidden_states_mem_ptr, final_hidden_states_mem_ptr, scratch);
        }
        return ret_env;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> moe_3gemm_swiglu_opt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_3gemm_fused_compressed>());
    return std::make_unique<moe_3gemm_swiglu_opt_impl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_3gemm_fused_compressed)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::moe_3gemm_swiglu_opt_impl)

#else

namespace ov::intel_gpu::ocl {

std::unique_ptr<primitive_impl> moe_3gemm_swiglu_opt::create_impl(const program_node& node, const RuntimeParams& params) const {
    OPENVINO_THROW("moe_3gemm_swiglu_opt depends on onednn.");
}

}  // namespace ov::intel_gpu::ocl

#endif
