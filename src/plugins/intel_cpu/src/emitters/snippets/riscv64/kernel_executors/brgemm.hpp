// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/rv64/brgemm/brgemm_types.hpp>
#include <cpu/rv64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "cache/multi_cache.h"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::intel_cpu::riscv64 {

struct BrgemmKernelConfig : public BrgemmGenericKernelConfig {
public:
    explicit BrgemmKernelConfig(const element::Type& input_type);

    void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) override;

    [[nodiscard]] std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::make_unique<BrgemmKernelConfig>(*this);
    }

    [[nodiscard]] size_t hash() const override {
        return m_hash;
    }

    [[nodiscard]] dnnl::impl::data_type_t get_input_type() const {
        return m_input_type;
    }

    [[nodiscard]] dnnl::impl::cpu::rv64::cpu_isa_t get_isa() const {
        return m_isa;
    }

    bool operator==(const BrgemmKernelConfig& rhs) const;
    bool operator!=(const BrgemmKernelConfig& rhs) const {
        return !(*this == rhs);
    }

private:
    [[nodiscard]] size_t compute_hash() const;

    dnnl::impl::data_type_t m_input_type{dnnl::impl::data_type::undef};
    dnnl::impl::cpu::rv64::cpu_isa_t m_isa{dnnl::impl::cpu::rv64::isa_undef};
    size_t m_hash{SIZE_MAX};
};

struct BrgemmCompiledKernel {
    std::shared_ptr<dnnl::impl::cpu::rv64::brgemm_kernel_t> kernel;
};

class BrgemmKernelExecutor : public CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
    };

    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);

    static void execute(const BrgemmKernelExecutor* executor, const call_args* args);

protected:
    [[nodiscard]] std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const BrgemmKernelConfig& config) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelConfig& config) const override;
};

}  // namespace ov::intel_cpu::riscv64
