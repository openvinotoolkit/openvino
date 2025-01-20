// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_batched.hpp"

#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "transformations/snippets/x64/op/gemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

BrgemmBatchedKernelConfig::BrgemmBatchedKernelConfig(const element::Type& in0_dtype,
                                       const element::Type& in1_dtype,
                                       size_t iter_count,
                                       bool is_with_comp,
                                       dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(),
      m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, is_with_comp, primitive_isa)),
      m_iter_count(iter_count) {
    m_hash = compute_hash();
}

BrgemmBatchedKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype,
                                               const element::Type& in1_dtype,
                                               bool is_with_comp,
                                               dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype, in1_dtype, primitive_isa, compute_hash(is_with_comp)),
      is_with_comp(is_with_comp) {}

bool BrgemmBatchedKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && is_with_comp == rhs.is_with_comp;
}

size_t BrgemmBatchedKernelConfig::StaticParams::compute_hash(bool is_with_comp) {
    return hash_combine(0, is_with_comp);
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmBatchedKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    ss << StaticBaseParams::to_string();
    ss << "is_with_comp = " << is_with_comp << "\n";
    return ss.str();
}
#endif

BrgemmBatchedKernelExecutor::BrgemmBatchedKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmBatchedKernelConfig config)
    : CPUKernelExecutor<BrgemmBatchedKernelConfig, BrgemmBatchedCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmBatchedCompiledKernel> BrgemmBatchedKernelExecutor::compile_kernel(const BrgemmBatchedKernelConfig& config) const {
    std::shared_ptr<BrgemmBatchedCompiledKernel> compiled_kernel = std::make_shared<BrgemmBatchedCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty()) {
        return compiled_kernel;
    }

    cpu::x64::brgemm_desc_t desc;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_desc_init(&desc,
                                               config.get_isa(),
                                               cpu::x64::brgemm_addr,
                                               config.get_dt_in0(),
                                               config.get_dt_in1(),
                                               false,
                                               false,
                                               cpu::x64::brgemm_row_major,
                                               1.f,
                                               config.get_beta(),
                                               config.get_LDA(),
                                               config.get_LDB(),
                                               config.get_LDC(),
                                               config.get_M(),
                                               config.get_N(),
                                               config.get_K(),
                                               nullptr) == dnnl_success,
                              "Cannot initialize brgemm descriptor due to invalid params");

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_kernel_create(&kernel_, desc) == dnnl_success,
                              "Cannot create brgemm kernel due to invalid params");
    compiled_kernel->brgemm_kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);

    return compiled_kernel;
}

void BrgemmBatchedKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmBatchedKernelConfig& config) const {
    return BrgemmBaseKernelExecutor::update_config(expr, linear_ir, config);
}

void BrgemmBatchedKernelExecutor::execute(const BrgemmBatchedKernelExecutor* executor, call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmBatchedKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    // Note: compensations should be applied only once, so we do it only on the first iteration, when beta == 0
    const auto is_with_comp = config.get_beta() == 0 && config.is_with_comp();
    execute_brgemm(kernel->brgemm_kernel, config.get_iter_count(), args->A, args->B, args->C, args->scratch, is_with_comp);
}

void BrgemmBatchedKernelExecutor::execute_brgemm(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                                                 size_t bs,
                                                 const void* pin0,
                                                 const void* pin1,
                                                 void* dst,
                                                 void* scratch,
                                                 bool with_comp) {
    cpu::x64::brgemm_kernel_params_t brgemm_p;
    brgemm_batch_element_t addr_batch;
    addr_batch.ptr.A = pin0;
    addr_batch.ptr.B = pin1;
    brgemm_p.batch = &addr_batch;
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = dst;
    brgemm_p.ptr_D = dst;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = with_comp;
    brgemm_p.do_apply_comp = with_comp;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr Brgemm kernel");
    (*kernel)(&brgemm_p);
}

}  // namespace intel_cpu
}  // namespace ov
