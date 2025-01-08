// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/brgemm/brgemm.hpp>
#include "libxsmm.h"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
// #include "emitters/snippets/jit_snippets_call_args.hpp"
// #include "openvino/core/type/element_type.hpp"
// #include "snippets/lowered/loop_info.hpp"
// #include "snippets/lowered/loop_manager.hpp"
#include "emitters/utils.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_base.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

struct BrgemmKernelConfig : public BrgemmBaseKernelConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype,
                       const element::Type& in1_dtype,
                       dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa = dnnl::impl::cpu::aarch64::cpu_isa_t::isa_undef);
    BrgemmKernelConfig() = delete;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelConfig>(new BrgemmKernelConfig(*this));
    }

    dnnl_data_type_t get_dt_in0() const {
        return get_static_params()->dt_in0;
    }
    dnnl_data_type_t get_dt_in1() const {
        return get_static_params()->dt_in1;
    }
    dnnl::impl::cpu::aarch64::cpu_isa_t get_isa() const {
        return m_static_params->isa;
    }
    libxsmm_bitfield get_static_compile_flags() const {
        return m_static_params->m_compile_flags;
    }
    libxsmm_bitfield get_compile_flags() const {
        return m_compile_flags;
    }
    void set_compile_flags(bool zero_beta) {
        if (zero_beta) {
            m_compile_flags = get_static_compile_flags() | LIBXSMM_GEMM_FLAG_BETA_0;
        } else {
            m_compile_flags = get_static_compile_flags();
        }
    }
    bool get_prefetching_flags() const {
        return m_static_params->m_prefetching_flags;
    }
    libxsmm_datatype get_type_in0() const {
        return m_static_params->m_type_in0;
    }
    libxsmm_datatype get_type_in1() const {
        return m_static_params->m_type_in1;
    }
    libxsmm_datatype get_type_out0() const {
        return m_static_params->m_type_out0;
    }
    libxsmm_datatype get_type_exec() const {
        return m_static_params->m_type_exec;
    }

private:
    struct StaticParams : public StaticBaseParams{
        StaticParams(const element::Type& in0_dtype,
                     const element::Type& in1_dtype,
                     dnnl::impl::cpu::aarch64::cpu_isa_t primitive_isa);
        virtual ~StaticParams() = default;

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }
        size_t compute_hash(dnnl::impl::cpu::aarch64::cpu_isa_t aarch_isa);

        dnnl::impl::cpu::aarch64::cpu_isa_t isa{dnnl::impl::cpu::aarch64::isa_undef};
        libxsmm_datatype m_type_in0;
        libxsmm_datatype m_type_in1;
        libxsmm_datatype m_type_out0;
        libxsmm_datatype m_type_exec;
        libxsmm_bitfield m_compile_flags {0};
        const bool m_prefetching_flags{false};
    };
    std::shared_ptr<StaticBaseParams> get_static_params() const override {
        return m_static_params;
    }

    libxsmm_bitfield m_compile_flags {0};
    std::shared_ptr<StaticParams> m_static_params{nullptr};
};

// The `update_kernel` method verifies that a compiled kernel is not nullptr.
// However, the compiled kernel might be empty in cases if nothing is to be compiled (`Config.is_empty() == true`).
// To cover this case, we wrap the `libxsmm_gemmfunction` in the separate structure which may contain empty
// `libxsmm_gemmfunction`
struct BrgemmTppCompiledKernel {
    std::shared_ptr<libxsmm_gemmfunction> brgemm_kernel = nullptr;
};

class BrgemmKernelExecutor : public BrgemmBaseKernelExecutor,
                             public CPUKernelExecutor<BrgemmKernelConfig, BrgemmTppCompiledKernel> {
public:
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);
    virtual ~BrgemmKernelExecutor() = default;

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmKernelExecutor* executor, void* in0, void* in1, void* out0);

private:
    std::shared_ptr<BrgemmTppCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

}  // namespace aarch64
}  // namespace intel_cpu
}  // namespace ov
