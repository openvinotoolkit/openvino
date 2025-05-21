// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <memory>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_base.hpp"

namespace ov::intel_cpu::x64 {

struct BrgemmAMXKernelConfig : public x64::BrgemmBaseKernelConfig {
public:
    BrgemmAMXKernelConfig(const element::Type& in0_dtype,
                          const element::Type& in1_dtype,
                          dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);
    BrgemmAMXKernelConfig() = delete;

    [[nodiscard]] std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::make_unique<BrgemmAMXKernelConfig>(*this);
    }

    [[nodiscard]] dnnl_dim_t get_inner_K_blk() const {
        return m_static_params->inner_k_blk;
    }
    [[nodiscard]] dnnl_dim_t get_vnni_factor() const {
        return m_static_params->vnni_factor;
    }

    [[nodiscard]] bool need_copy_a(dnnl_dim_t K) const;

private:
    struct StaticParams : StaticBaseParams {
        StaticParams(const element::Type& in0_dtype,
                     const element::Type& in1_dtype,
                     dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);

        const dnnl_dim_t inner_k_blk{0};
        const dnnl_dim_t vnni_factor{0};

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }
#ifdef SNIPPETS_DEBUG_CAPS
        [[nodiscard]] std::string to_string() const override;
#endif
    private:
        static size_t compute_hash(dnnl_dim_t inner_k_blk, dnnl_dim_t vnni_factor);
    };

    [[nodiscard]] std::shared_ptr<StaticBaseParams> get_static_params() const override {
        return m_static_params;
    }

    std::shared_ptr<StaticParams> m_static_params{nullptr};
};

struct BrgemmAMXCompiledKernel {
    struct BrgemmKernel {
        std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_kernel{nullptr};
        // Note: Palette is treated as a part of a kernel because it is initialized during the kernel compilation stage.
        //       Each kernel need to store the pallet it was compiled with.
        char palette[64] = {};
    };

    std::shared_ptr<BrgemmKernel> K_body_kernel{nullptr};
    std::shared_ptr<BrgemmKernel> K_tail_kernel{nullptr};
    std::shared_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> brgemm_copy_a_kernel{nullptr};
};

class BrgemmAMXKernelExecutor : public BrgemmBaseKernelExecutor,
                                public CPUKernelExecutor<BrgemmAMXKernelConfig, BrgemmAMXCompiledKernel> {
public:
    struct call_args {
        const uint8_t* A = nullptr;
        const uint8_t* B = nullptr;
        void* C = nullptr;
        uint8_t* scratch = nullptr;
        amx_tile_config_t* amx_tile_config = nullptr;
    };
    BrgemmAMXKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmAMXKernelConfig config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmAMXKernelExecutor* executor, call_args* args);

protected:
    [[nodiscard]] std::shared_ptr<BrgemmAMXCompiledKernel> compile_kernel(
        const BrgemmAMXKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmAMXKernelConfig& config) const override;

    static void configure_tiles_if_needed(amx_tile_config_t* config,
                                          const char* palette,
                                          dnnl_dim_t M,
                                          dnnl_dim_t N,
                                          dnnl_dim_t K);

    static void create_brgemm_copy_a_kernel(
        std::shared_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& kernel,
        dnnl::impl::cpu::x64::cpu_isa_t isa,
        dnnl_data_type_t dt,
        dnnl_dim_t K,
        dnnl_dim_t K_blk,
        dnnl_dim_t K_tail,
        dnnl_dim_t src_stride,
        dnnl_dim_t LDA);

    static void execute_brgemm_copy_a_kernel(
        const std::shared_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& kernel,
        const void* src,
        const void* tr_src,
        dnnl_dim_t M,
        dnnl_dim_t K);
};
#define GET_OFF_BRGEMM_AMX_ARGS(field) offsetof(BrgemmAMXKernelExecutor::call_args, field)

}  // namespace ov::intel_cpu::x64
