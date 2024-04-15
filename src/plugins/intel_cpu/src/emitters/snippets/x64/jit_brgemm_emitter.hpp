// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_kernel_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>

namespace ov {
namespace intel_cpu {
class BrgemmKernelExecutor;
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
    friend BrgemmKernelExecutor;
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta, bool is_with_amx,
                       size_t M = 0, size_t N = 0, size_t K = 0,
                       size_t LDA = 0, size_t LDB = 0, size_t LDC = 0);
    BrgemmKernelConfig() = default;
    bool is_complete() const override;
    size_t hash() const;
    bool operator==(const BrgemmKernelConfig& rhs) const;
    bool operator!=(const BrgemmKernelConfig& rhs) const;
private:
    dnnl_data_type_t dt_in0 {dnnl_f32}, dt_in1 {dnnl_f32};
    char palette[64] = {};
    bool is_with_amx {false};
    bool is_with_comp {false};
    float beta {0};
    dnnl::impl::cpu::x64::cpu_isa_t isa {dnnl::impl::cpu::x64::isa_undef};
    dnnl_dim_t M {0}, N {0}, K {0}, LDA {0}, LDB {0}, LDC {0};
};


class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, dnnl::impl::cpu::x64::brgemm_kernel_t> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        void* scratch = nullptr;
        amx_tile_config_t* amx_tile_config = nullptr;
    };
    BrgemmKernelExecutor(ov::intel_cpu::MultiCachePtr kernel_cache, const std::shared_ptr<BrgemmKernelConfig>& config);
    static void execute(const BrgemmKernelExecutor* desc, call_args* args);
    void update_kernel_config(size_t M, size_t N, size_t K,  size_t LDA, size_t LDB, size_t LDC);
protected:
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> compile_kernel(const std::shared_ptr<BrgemmKernelConfig>& c) const override;
};

class jit_brgemm_emitter : public jit_emitter {
public:
    jit_brgemm_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr,
                       const snippets::KernelExecutorTablePtr& kernel_table,
                       const ov::intel_cpu::MultiCachePtr& compiled_kernel_cache);

    size_t get_inputs_num() const override { return m_with_scratch ? 3 : 2; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

    static size_t get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
    static size_t get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_brgemm_kernel_call(Xbyak::Reg64 addr_A, Xbyak::Reg64 addr_B, Xbyak::Reg64 scratch, Xbyak::Reg64 addr_C,
                                 size_t in0_kernel_offset = 0, size_t in1_kernel_offset = 0,
                                 size_t in2_kernel_offset = 0, size_t out0_kernel_offset = 0) const;

    bool m_with_scratch = false;

    size_t m_load_offset_a = 0lu;
    size_t m_load_offset_b = 0lu;
    size_t m_load_offset_scratch = 0lu;
    size_t m_store_offset_c = 0lu;
    std::shared_ptr<BrgemmKernelExecutor> m_kernel_executor = nullptr;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_brgemm_emitter(const jit_brgemm_emitter *emitter);
#endif
};

}   // namespace intel_cpu
}   // namespace ov
