// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include <memory>

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::x64 {

BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype,
                                       const element::Type& in1_dtype,
                                       bool is_with_comp,
                                       dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(),
      m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, is_with_comp, primitive_isa)) {
    m_hash = compute_hash();
}

BrgemmKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype,
                                               const element::Type& in1_dtype,
                                               bool is_with_comp,
                                               dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype, in1_dtype, primitive_isa, compute_hash(is_with_comp)),
      is_with_comp(is_with_comp) {}

bool BrgemmKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && is_with_comp == rhs.is_with_comp;
}

size_t BrgemmKernelConfig::StaticParams::compute_hash(bool is_with_comp) {
    return hash_combine(0, is_with_comp);
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    ss << StaticBaseParams::to_string();
    ss << "is_with_comp = " << is_with_comp << "\n";
    return ss.str();
}
#endif

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config)
    : CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    std::shared_ptr<BrgemmCompiledKernel> compiled_kernel = std::make_shared<BrgemmCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty()) {
        return compiled_kernel;
    }

    create_brgemm_kernel(compiled_kernel->brgemm_kernel,
                         config.get_dt_in0(),
                         config.get_dt_in1(),
                         config.get_isa(),
                         config.get_M(),
                         config.get_N(),
                         config.get_K(),
                         config.get_LDA(),
                         config.get_LDB(),
                         config.get_LDC(),
                         config.get_beta());

    return compiled_kernel;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    return BrgemmBaseKernelExecutor::update_config(expr, linear_ir, config);
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    // Note: compensations should be applied only once, so we do it only on the first iteration, when beta == 0
    const auto is_with_comp = config.get_beta() == 0 && config.is_with_comp();
    execute_brgemm_kernel(kernel->brgemm_kernel, args->A, args->B, args->C, args->scratch, is_with_comp);
}

#ifdef SNIPPETS_DEBUG_CAPS
BrgemmKernelReferenceExecutor::BrgemmKernelReferenceExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                                                             BrgemmKernelConfig config)
    : BrgemmKernelExecutor(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelReferenceExecutor::compile_kernel(const BrgemmKernelConfig& c) const {
    const auto& res = std::make_shared<BrgemmCompiledKernel>();
    res->brgemm_kernel = std::make_shared<brgemm_ref_kernel>(c);
    return res;
}

brgemm_ref_kernel::brgemm_ref_kernel(BrgemmKernelConfig c) : m_config(std::move(c)) {
    OV_CPU_JIT_EMITTER_ASSERT(!m_config.is_with_comp(), "brgemm_ref_kernel doesn't currently support compensations");
    OV_CPU_JIT_EMITTER_ASSERT(
        m_config.get_dt_in0() == m_config.get_dt_in1() && m_config.get_dt_in0() == dnnl_data_type_t::dnnl_f32,
        "brgemm_ref_kernel currently supports only fp32 inputs");
}

void brgemm_ref_kernel::operator()(dnnl::impl::cpu::x64::brgemm_kernel_params_t* args) const {
    auto A = reinterpret_cast<const float*>(args->ptr_A);
    auto B = reinterpret_cast<const float*>(args->ptr_B);
    auto C = reinterpret_cast<float*>(args->ptr_C);
    for (dnnl_dim_t m = 0; m < m_config.get_M(); m++) {
        for (dnnl_dim_t n = 0; n < m_config.get_N(); n++, B++) {
            C[n] = 0;
            for (dnnl_dim_t k = 0; k < m_config.get_K(); k++) {
                C[n] += A[k] * B[k * m_config.get_LDB()];
            }
        }
        B -= m_config.get_N();
        A += m_config.get_LDA();
        C += m_config.get_LDC();
    }
}
#endif

}  // namespace ov::intel_cpu::x64
