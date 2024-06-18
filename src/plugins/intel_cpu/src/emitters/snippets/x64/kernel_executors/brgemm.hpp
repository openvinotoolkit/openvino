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
class BrgemmKernelExecutor;
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
    friend BrgemmKernelExecutor;

public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta,
                       bool is_with_amx, bool is_with_comp,
                       size_t M = 0, size_t N = 0, size_t K = 0,
                       size_t LDA = 0, size_t LDB = 0, size_t LDC = 0);
    BrgemmKernelConfig() = default;
    bool is_completed() const override;
    size_t hash() const override { return m_hash; }
    std::shared_ptr<GenericConfig> clone() const override {
        return std::make_shared<BrgemmKernelConfig>(*this);
    }
#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

private:
    size_t compute_hash() const;
    dnnl_data_type_t dt_in0 {dnnl_f32}, dt_in1 {dnnl_f32};
    bool is_with_amx {false};
    bool is_with_comp {false};
    float beta {0};
    dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::isa_undef};
    dnnl_dim_t M {0}, N {0}, K {0}, LDA {0}, LDB {0}, LDC {0};
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
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                         const std::shared_ptr<BrgemmKernelConfig>& config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmKernelExecutor* desc, call_args* args);

protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const std::shared_ptr<const BrgemmKernelConfig>& c) const override;
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr, std::shared_ptr<BrgemmKernelConfig>& config) const override;
};
}   // namespace intel_cpu
}   // namespace ov
