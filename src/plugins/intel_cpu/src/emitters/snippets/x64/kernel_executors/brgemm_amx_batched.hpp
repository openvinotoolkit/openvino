// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>

#include "brgemm_base.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

namespace ov::intel_cpu {

struct BrgemmAMXBatchedKernelConfig : public x64::BrgemmBaseKernelConfig {
public:
    BrgemmAMXBatchedKernelConfig(const element::Type& in0_dtype,
                                 const element::Type& in1_dtype,
                                 size_t iter_count,
                                 dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);
    BrgemmAMXBatchedKernelConfig() = delete;

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmAMXBatchedKernelConfig>(new BrgemmAMXBatchedKernelConfig(*this));
    }

    dnnl_dim_t get_inner_K_blk() const {
        return m_static_params->inner_k_blk;
    }
    dnnl_dim_t get_vnni_factor() const {
        return m_static_params->vnni_factor;
    }

    size_t get_iter_count() const {
        return m_iter_count;
    }

    bool need_copy_a(dnnl_dim_t K) const;

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
        std::string to_string() const override;
#endif
    private:
        static size_t compute_hash(dnnl_dim_t inner_k_blk, dnnl_dim_t vnni_factor);
    };

    std::shared_ptr<StaticBaseParams> get_static_params() const override {
        return m_static_params;
    }

    std::shared_ptr<StaticParams> m_static_params{nullptr};
    size_t m_iter_count{1};
};

struct BrgemmAMXBatchedCompiledKernel {
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

class BrgemmAMXBatchedKernelExecutor
    : public x64::BrgemmBaseKernelExecutor,
      public CPUKernelExecutor<BrgemmAMXBatchedKernelConfig, BrgemmAMXBatchedCompiledKernel> {
public:
    struct call_args {
        const uint8_t* A = nullptr;
        const uint8_t* B = nullptr;
        void* C = nullptr;
        uint8_t* scratch = nullptr;
        amx_tile_config_t* amx_tile_config = nullptr;
    };
    BrgemmAMXBatchedKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmAMXBatchedKernelConfig config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmAMXBatchedKernelExecutor* executor, call_args* args);

protected:
    std::shared_ptr<BrgemmAMXBatchedCompiledKernel> compile_kernel(
        const BrgemmAMXBatchedKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmAMXBatchedKernelConfig& config) const override;

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

    static void execute_brgemm(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                               size_t bs,
                               size_t stride_A,
                               size_t stride_B,
                               const void* pin0,
                               const void* pin1,
                               void* dst,
                               void* scratch,
                               bool with_comp);
};

}  // namespace ov::intel_cpu
