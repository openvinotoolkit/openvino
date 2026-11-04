// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include <cpu/rv64/brgemm/brgemm.hpp>
#include <cpu/rv64/brgemm/brgemm_types.hpp>
#include <cpu/rv64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "cache/multi_cache.h"
#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::riscv64 {

namespace {

dnnl::impl::data_type_t get_dnnl_type(const element::Type& type) {
    if (type == element::f32) {
        return dnnl::impl::data_type::f32;
    }
    if (type == element::bf16) {
        return dnnl::impl::data_type::bf16;
    }
    if (type == element::f16) {
        return dnnl::impl::data_type::f16;
    }
    OV_CPU_JIT_EMITTER_THROW("Unsupported BrgemmCPU input type: ", type);
}

dnnl::impl::cpu::rv64::cpu_isa_t get_brgemm_isa(dnnl::impl::data_type_t type) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu::rv64;
    if (type == data_type::bf16) {
        return zvfbfwma;
    }
    if (type == data_type::f16) {
        return zvfh;
    }
    return v;
}

}  // namespace

BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& input_type)
    : m_input_type(get_dnnl_type(input_type)),
      m_isa(get_brgemm_isa(m_input_type)),
      m_hash(compute_hash()) {}

void BrgemmKernelConfig::update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) && m_input_type == rhs.m_input_type && m_isa == rhs.m_isa;
}

size_t BrgemmKernelConfig::compute_hash() const {
    auto seed = BrgemmGenericKernelConfig::compute_hash();
    seed = dnnl::impl::hash_combine(seed, m_input_type);
    return dnnl::impl::hash_combine(seed, m_isa);
}

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config)
    : CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    auto compiled = std::make_shared<BrgemmCompiledKernel>();
    if (config.is_empty()) {
        return compiled;
    }

    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu::rv64;

    brgemm_desc_t desc{};
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_init(&desc,
                                               config.get_isa(),
                                               brgemm_strd,
                                               config.get_input_type(),
                                               config.get_input_type(),
                                               brgemm_col_major,
                                               1.F,
                                               config.get_beta(),
                                               config.get_LDB(),
                                               config.get_LDA(),
                                               config.get_LDC(),
                                               config.get_N(),
                                               config.get_M(),
                                               config.get_K()) == status::success,
                              "Cannot initialize RV64 brgemm descriptor");

    brgemm_kernel_t* raw_kernel = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_kernel_create(&raw_kernel, desc) == status::success,
                              "Cannot create RV64 brgemm kernel");
    compiled->kernel = std::shared_ptr<brgemm_kernel_t>(raw_kernel, brgemm_kernel_destroy);
    return compiled;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    const auto [M, N, K, beta, LDC] = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);
    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDB = snippets::utils::get_dim_stride(expr->get_input_port(1));
    config.update(M, N, K, LDA, LDB, LDC, beta);
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, const call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "Brgemm executor is null");
    OV_CPU_JIT_EMITTER_ASSERT(args, "Brgemm call arguments are null");

    const auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel && kernel->kernel, "RV64 brgemm kernel is null");

    dnnl::impl::cpu::rv64::brgemm_kernel_execute(kernel->kernel.get(),
                                                 args->B,
                                                 args->A,
                                                 args->C,
                                                 config.get_M(),
                                                 config.get_beta());
}

}  // namespace ov::intel_cpu::riscv64
