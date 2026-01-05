// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "micro_utils.hpp"
#include "moe_3gemm_base.hpp"
#include "moe_gemm_gen_opt.hpp"
#include "moe_gemm_inst.h"
#include "ocl_v2/utils/jitter.hpp"
using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {
#ifdef ENABLE_ONEDNN_FOR_GPU
#    include "micro_utils.hpp"

class MoE3GemmMicroGenerator : public MoEGemmOptGeneratorBase {
public:
    explicit MoE3GemmMicroGenerator(MoE3GemmMicroKernelType type)
        : MoEGemmOptGeneratorBase("moe_gemm",
                                  type == MoE3GemmMicroKernelType::MLP_GATE ? "_prefill_mlp_gate"
                                  : type == MoE3GemmMicroKernelType::MLP_UP ? "_prefill_mlp_up"
                                                                            : "_prefill_mlp_down"),
          m_type(type) {
        switch (m_type) {
        case MoE3GemmMicroKernelType::MLP_GATE:
            m_wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_0);
            m_scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_0);
            m_zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_0);
            break;
        case MoE3GemmMicroKernelType::MLP_UP:
            m_wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_1);
            m_scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_1);
            m_zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_1);
            break;
        case MoE3GemmMicroKernelType::MLP_DOWN:
            m_wei_idx = static_cast<int>(MOE3GemmInputIndex::WEIGHT_2);
            m_scale_idx = static_cast<int>(MOE3GemmInputIndex::SCALE_2);
            m_zp_idx = static_cast<int>(MOE3GemmInputIndex::ZP_2);
            break;
        default:
            OPENVINO_THROW("Unsupported MoE3GemmMicroKernelType");
            break;
        }
    }

    [[nodiscard]] std::string get_build_options(const kernel_impl_params& params) const override;

    [[nodiscard]] KernelData get_kernel_data(const kernel_impl_params& params) const override;

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        OPENVINO_THROW("Use overloaded version instead");
    }
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params, const micro::Package& moe_gemm, const moe_3gemm_config& cfg) const;

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override;

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

    static const moe_3gemm_config get_moe_3gemm_cfg(const kernel_impl_params& params) {
        moe_3gemm_config cfg;
        auto desc = params.typed_desc<moe_3gemm_fused_compressed>();
        cfg.weight_group_size = static_cast<int32_t>(desc->_config.group_size);
        cfg.has_batch_dim = desc->_config.has_batch_dim;
        return cfg;
    }

    static void init_microkernels(const kernel_impl_params& params, micro::Package& gemm_moe, MoE3GemmMicroKernelType type) noexcept;
    MoE3GemmMicroKernelType m_type;
    int m_wei_idx;
    int m_scale_idx;
    int m_zp_idx;
    static std::mutex mtx;

    struct GemmCacheKey {
        ov::Shape weight_shape;
        ov::element::Type weight_dt;

        ov::Shape scale_shape;
        ov::element::Type scale_dt;

        ov::Shape zp_shape;
        ov::element::Type zp_dt;

        bool operator==(const GemmCacheKey& other) const {
            return weight_shape == other.weight_shape && weight_dt == other.weight_dt && scale_shape == other.scale_shape && scale_dt == other.scale_dt &&
                   zp_shape == other.zp_shape && zp_dt == other.zp_dt;
        }
    };

    struct GemmCacheKeyHash {
        size_t operator()(const GemmCacheKey& k) const noexcept {
            size_t h = 0;

            auto hash_combine = [](size_t& seed, size_t v) {
                seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };

            auto hash_shape = [&](const ov::Shape& s) {
                for (auto v : s) {
                    hash_combine(h, std::hash<size_t>()(v));
                }
            };

            hash_shape(k.weight_shape);
            hash_shape(k.scale_shape);
            hash_shape(k.zp_shape);

            hash_combine(h, std::hash<std::string>()(k.weight_dt.to_string()));
            hash_combine(h, std::hash<std::string>()(k.scale_dt.to_string()));
            hash_combine(h, std::hash<std::string>()(k.zp_dt.to_string()));
            return h;
        }
    };

    static std::unordered_map<GemmCacheKey, micro::Package, GemmCacheKeyHash> s_gemm_cache;
};
#endif
}  // namespace ov::intel_gpu::ocl
