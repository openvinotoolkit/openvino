// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>

namespace ov {
namespace intel_cpu {
struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta,
                       bool is_with_amx, bool is_with_comp);
    BrgemmKernelConfig() = default;
    bool is_completed() const override;
    size_t hash() const override { return m_hash; }
    bool operator==(const BrgemmKernelConfig& rhs) const;
    bool operator!=(const BrgemmKernelConfig& rhs) const {return !(*this == rhs);}
    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelConfig>( new BrgemmKernelConfig(*this));
    }
    void update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC);

    dnnl_data_type_t get_dt_in0() const { return m_static_params->dt_in0; }
    dnnl_data_type_t get_dt_in1() const { return m_static_params->dt_in1; }

    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const { return m_static_params->isa; }
    bool is_with_amx() const {return m_static_params->is_with_amx; }
    bool is_with_comp() const { return m_static_params->is_with_comp; }
    float get_beta() const { return m_static_params->beta; }

    dnnl_dim_t get_M() const { return m_M; }
    dnnl_dim_t get_N() const { return m_N; }
    dnnl_dim_t get_K() const { return m_K; }

    dnnl_dim_t get_LDA() const { return m_LDA; }
    dnnl_dim_t get_LDB() const { return m_LDB; }
    dnnl_dim_t get_LDC() const { return m_LDC; }

    explicit operator amx_tile_config_t() const;
    inline bool compatible(amx_tile_config_t* rhs) const {
        return rhs && rhs->M == m_M && rhs->N == m_N && rhs->K == m_K;
    }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

private:
    struct StaticParams {
        StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta,
                            bool is_with_amx, bool is_with_comp);
        const dnnl_data_type_t dt_in0 {dnnl_f32}, dt_in1 {dnnl_f32};
        const float beta {0};
        const bool is_with_amx {false};
        const bool is_with_comp {false};
        const dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::isa_undef};
        const size_t hash{0};
        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {return !(*this == rhs);}
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif
    };
    size_t compute_hash() const;
    std::shared_ptr<StaticParams> m_static_params;
    dnnl_dim_t m_M {0}, m_N {0}, m_K {0}, m_LDA {0}, m_LDB {0}, m_LDC {0};
    size_t m_hash {SIZE_MAX};
};

struct BrgemmCompiledKernel {
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> compiled_kernel = nullptr;
    // Note: Palette is treated as a part of a kernel because it is initialized during the kernel compilation stage.
    //       Each kernel need to store the pallet it was compiled with.
    char palette[64] = {};
};

class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        void* scratch = nullptr;
        amx_tile_config_t* amx_tile_config = nullptr;
    };
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmKernelExecutor* executor, call_args* args);

protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const override;
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr, BrgemmKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)
}   // namespace intel_cpu
}   // namespace ov
