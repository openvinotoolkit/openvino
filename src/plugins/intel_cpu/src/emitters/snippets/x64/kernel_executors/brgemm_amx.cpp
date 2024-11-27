// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_amx.hpp"

#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/amx_tile_configure.hpp>


#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;


namespace ov {
namespace intel_cpu {

BrgemmAMXKernelConfig::BrgemmAMXKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(), m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, primitive_isa)) {
    m_hash = compute_hash();
}

BrgemmAMXKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                                  dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype, in1_dtype, primitive_isa), inner_k_blk(DIM_CAST(brgemm_utils::repacking::compute_inner_k_block(in0_dtype))),
      vnni_factor(DIM_CAST(brgemm_utils::compute_vnni_factor(in0_dtype))), m_hash(compute_hash()) {}

bool BrgemmAMXKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && inner_k_blk == rhs.inner_k_blk && vnni_factor == rhs.vnni_factor;
}

size_t BrgemmAMXKernelConfig::StaticParams::compute_hash() const {
    size_t seed = StaticBaseParams::compute_hash();
    seed = hash_combine(seed, inner_k_blk);
    return hash_combine(seed, vnni_factor);
}

bool BrgemmAMXKernelConfig::need_copy_a(dnnl_dim_t K) const {
    return K % get_vnni_factor() > 0;
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmAMXKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    ss << StaticBaseParams::to_string();
    ss << "inner_k_blk = " << inner_k_blk << "\n";
    ss << "vnni_factor = " << vnni_factor << "\n";
    return ss.str();
}
#endif

BrgemmAMXKernelExecutor::BrgemmAMXKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmAMXKernelConfig config) :
        CPUKernelExecutor<BrgemmAMXKernelConfig, BrgemmAMXCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmAMXCompiledKernel> BrgemmAMXKernelExecutor::compile_kernel(const BrgemmAMXKernelConfig& config) const {
    std::shared_ptr<BrgemmAMXCompiledKernel> compiled_kernel = std::make_shared<BrgemmAMXCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty())
        return compiled_kernel;

    auto K_tail = config.get_K() % config.get_inner_K_blk();
    auto K_body = config.get_K() - K_tail;

    float beta = config.get_beta();

    if (K_body != 0) {
        create_brgemm_kernel(compiled_kernel->brgemm_kernel_k_body, config.get_dt_in0(), config.get_dt_in1(), config.get_isa(),
                             config.get_M(), config.get_N(), K_body, config.get_LDA(), config.get_LDB(), config.get_LDC(), beta,
                             true, compiled_kernel->palette_body);
        beta = 1;
    }

    if (K_tail != 0) {
        auto LDA = config.get_LDA();
        if (config.need_copy_a(K_tail)) {
            const auto copy_A_src_stride = LDA * dnnl_data_type_size(config.get_dt_in0());
            K_tail = ov::snippets::utils::rnd_up(K_tail, config.get_vnni_factor());
            LDA = K_tail;

            create_brgemm_copy_a_kernel(compiled_kernel->brgemm_copy_a_kernel, config.get_isa(), config.get_dt_in0(),
                                        config.get_K(), config.get_inner_K_blk(), K_tail, copy_A_src_stride, LDA);
        }

        create_brgemm_kernel(compiled_kernel->brgemm_kernel_k_tail, config.get_dt_in0(), config.get_dt_in1(), config.get_isa(),
                             config.get_M(), config.get_N(), K_tail, LDA, config.get_LDB(), config.get_LDC(), beta,
                             true, compiled_kernel->palette_tail);
    }

    return compiled_kernel;
}

void BrgemmAMXKernelExecutor::create_brgemm_copy_a_kernel(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& kernel,
                                                          dnnl::impl::cpu::x64::cpu_isa_t isa, dnnl_data_type_t dt,
                                                          dnnl_dim_t K, dnnl_dim_t K_blk, dnnl_dim_t K_tail, dnnl_dim_t src_stride, dnnl_dim_t LDA) {
    matmul::brgemm_matmul_conf_t conf_;
    conf_.src_tag = dnnl_abcd; // unused
    conf_.K = K;
    conf_.K_tail = K_tail;
    conf_.K_blk = K_blk;
    conf_.use_buffer_a_tail_only = false;
    conf_.LDA = LDA;
    conf_.has_zero_point_b = false;
    conf_.s8s8_compensation_required = false;
    conf_.wei_zp_type = dnnl::impl::cpu::x64::none;
    conf_.src_zp_type = dnnl::impl::cpu::x64::none;
    conf_.src_dt = dt;
    conf_.copy_A_src_stride = src_stride;
    conf_.a_dt_sz = dnnl_data_type_size(conf_.src_dt);
    // copied A has the same precision of original
    conf_.tr_a_dt_sz = dnnl_data_type_size(conf_.src_dt);
    conf_.transposed_A = false;
    conf_.isa = isa;

    OV_CPU_JIT_EMITTER_ASSERT(create_brgemm_matmul_copy_a(kernel, &conf_) == dnnl_success,
                              "Cannot create brgemm copy a kernel due to invalid params");
}

void BrgemmAMXKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                            const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                            BrgemmAMXKernelConfig& config) const {
    return BrgemmBaseKernelExecutor::update_config(expr, linear_ir, config);
}

void BrgemmAMXKernelExecutor::configure_tiles_if_needed(amx_tile_config_t* config, const char* palette, dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K) {
    auto compatible = [&](amx_tile_config_t* rhs) {
        return rhs && rhs->M == M && rhs->N == N && rhs->K == K;
    };
    if (config && !compatible(config)) {
        config->M = M; config->N = N; config->K = K;
        cpu::x64::amx_tile_configure(palette);
    }
}

void BrgemmAMXKernelExecutor::execute_brgemm_copy_a_kernel(const std::unique_ptr<matmul::jit_brgemm_matmul_copy_a_t>& kernel,
                                                           const void* src, const void* tr_src, dnnl_dim_t M, dnnl_dim_t K) {
    auto ctx = matmul::jit_brgemm_matmul_copy_a_t::ctx_t();

    ctx.current_M_blk = M;
    ctx.zp_b_compensation_buffer_ptr = nullptr;
    ctx.zp_a_compensation_result_ptr = nullptr;
    ctx.zp_b_neg_value_ptr = nullptr;
    ctx.zp_ab_comp_ptr = nullptr;
    ctx.src = src;
    ctx.tr_src = tr_src;
    ctx.current_K_start = 0;
    ctx.current_K_blk = K;

    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr brgemm_copy_a_kernel");
    (*kernel)(&ctx);
}

void BrgemmAMXKernelExecutor::execute(const BrgemmAMXKernelExecutor* executor, call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmAMXKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    const auto* src_ptr = args->A;
    const auto* wei_ptr = args->B;
    auto* scratch = args->scratch;

    const auto K_tail = config.get_K() % config.get_inner_K_blk();
    const auto K_body = config.get_K() - K_tail;

    if (K_body != 0) {
        configure_tiles_if_needed(args->amx_tile_config, kernel->palette_body, config.get_M(), config.get_N(), K_body);
        execute_brgemm_kernel(kernel->brgemm_kernel_k_body, src_ptr, wei_ptr, args->C, scratch, false);

        src_ptr = src_ptr + K_body * dnnl_data_type_size(config.get_dt_in0());
        wei_ptr = wei_ptr + (K_body * config.get_LDB()) * dnnl_data_type_size(config.get_dt_in1());
    }

    if (K_tail != 0) {
        if (config.need_copy_a(K_tail)) {
            auto* tr_src = scratch + BrgemmCPU::SCRATCH_BYTE_SIZE;

            execute_brgemm_copy_a_kernel(kernel->brgemm_copy_a_kernel, src_ptr, tr_src, config.get_M(), K_tail);
            src_ptr = tr_src;
        }

        configure_tiles_if_needed(args->amx_tile_config, kernel->palette_tail, config.get_M(), config.get_N(), K_tail);
        execute_brgemm_kernel(kernel->brgemm_kernel_k_tail, src_ptr, wei_ptr, args->C, scratch, false);
    }
}

}   // namespace intel_cpu
}   // namespace ov
