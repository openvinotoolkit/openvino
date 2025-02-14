// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/utils.hpp"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "libxsmm.h"

namespace ov::intel_cpu::tpp {

struct BrgemmKernelConfig : public BrgemmGenericKernelConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype);
    BrgemmKernelConfig() = delete;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelConfig>(new BrgemmKernelConfig(*this));
    }

    bool operator==(const BrgemmKernelConfig& rhs) const;
    bool operator!=(const BrgemmKernelConfig& rhs) const {
        return !(*this == rhs);
    }

    void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) override;

    size_t hash() const override {
        return m_hash;
    }
    size_t compute_hash() const;

    libxsmm_bitfield get_compile_flags() const {
        return m_compile_flags;
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
    libxsmm_bitfield get_static_compile_flags() const {
        return m_static_params->m_compile_flags;
    }
#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override;
#endif

private:
    struct StaticParams {
        StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype);

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }
        size_t hash() const {
            return m_hash;
        }
        size_t compute_hash();

#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif

        libxsmm_datatype m_type_in0;
        libxsmm_datatype m_type_in1;
        libxsmm_datatype m_type_out0;
        libxsmm_datatype m_type_exec;
        libxsmm_bitfield m_compile_flags;
        bool m_prefetching_flags;

        size_t m_hash{SIZE_MAX};
    };

    const std::shared_ptr<StaticParams>& get_static_params() const {
        return m_static_params;
    }
    void set_compile_flags(const libxsmm_bitfield& compile_flags) {
        m_compile_flags = compile_flags;
    }
    libxsmm_bitfield m_compile_flags{0};
    std::shared_ptr<StaticParams> m_static_params{nullptr};

    size_t m_hash{SIZE_MAX};
};

// The `update_kernel` method verifies that a compiled kernel is not nullptr.
// However, the compiled kernel might be empty in cases if nothing is to be compiled (`Config.is_empty() == true`).
// To cover this case, we wrap the `libxsmm_gemmfunction` in the separate structure which may contain empty
// `libxsmm_gemmfunction`
struct BrgemmTppCompiledKernel {
    std::shared_ptr<libxsmm_gemmfunction> brgemm_kernel = nullptr;
};

class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, BrgemmTppCompiledKernel> {
public:
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);

    // Function that will be called in runtime to execute the kernel
    static void execute(const BrgemmKernelExecutor* executor, void* in0, void* in1, void* out0);

private:
    std::shared_ptr<BrgemmTppCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

}  // namespace ov::intel_cpu::tpp
