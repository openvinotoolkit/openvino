// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_amx.hpp"

#include <cpu/x64/amx_tile_configure.hpp>

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

#define INNER_K_BLK(dtype) static_cast<dnnl_dim_t>((brgemm_utils::repacking::compute_inner_k_block(in0_dtype)))
#define VNNI_FACTOR(dtype) static_cast<dnnl_dim_t>((brgemm_utils::compute_vnni_factor(in0_dtype)))
#define EQ(X)              X == rhs.X
#define HASH(X)            seed = hash_combine(seed, X)

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::x64 {

BrgemmAMXKernelConfig::BrgemmAMXKernelConfig(const element::Type& in0_dtype,
                                             const element::Type& in1_dtype,
                                             dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(),
      m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, primitive_isa)) {
    m_hash = compute_hash();
}

BrgemmAMXKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype,
                                                  const element::Type& in1_dtype,
                                                  dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype,
                       in1_dtype,
                       primitive_isa,
                       compute_hash(INNER_K_BLK(in0_dtype), VNNI_FACTOR(in0_dtype))),
      inner_k_blk(INNER_K_BLK(in0_dtype)),
      vnni_factor(VNNI_FACTOR(in0_dtype)) {}

bool BrgemmAMXKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && EQ(inner_k_blk) && EQ(vnni_factor);
}

size_t BrgemmAMXKernelConfig::StaticParams::compute_hash(dnnl_dim_t inner_k_blk, dnnl_dim_t vnni_factor) {
    size_t seed = 0;
    HASH(inner_k_blk);
    HASH(vnni_factor);
    return seed;
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

BrgemmAMXKernelExecutor::BrgemmAMXKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                                                 BrgemmAMXKernelConfig config)
    : CPUKernelExecutor<BrgemmAMXKernelConfig, BrgemmAMXCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

namespace {
struct BrgemmCopyAKey {
    BrgemmCopyAKey(cpu_isa_t isa,
                   dnnl_data_type_t dt,
                   dnnl_dim_t K,
                   dnnl_dim_t K_blk,
                   dnnl_dim_t K_tail,
                   dnnl_dim_t src_stride,
                   dnnl_dim_t LDA)
        : isa(isa),
          dt(dt),
          K{K},
          K_blk{K_blk},
          K_tail{K_tail},
          src_stride{src_stride},
          LDA{LDA} {}

    [[nodiscard]] size_t hash() const {
        size_t seed = 0;
        HASH(isa);
        HASH(dt);
        HASH(K);
        HASH(K_blk);
        HASH(K_tail);
        HASH(src_stride);
        HASH(LDA);
        return seed;
    }
    bool operator==(const BrgemmCopyAKey& rhs) const {
        return EQ(isa) && EQ(dt) && EQ(K) && EQ(K_blk) && EQ(K_tail) && EQ(src_stride) && EQ(LDA);
    }

    cpu_isa_t isa{cpu_isa_t::isa_undef};
    dnnl_data_type_t dt{dnnl_data_type_t::dnnl_data_type_undef};
    dnnl_dim_t K{0}, K_blk{0}, K_tail{0}, src_stride{0}, LDA{0};
};
}  // namespace

std::shared_ptr<BrgemmAMXCompiledKernel> BrgemmAMXKernelExecutor::compile_kernel(
    const BrgemmAMXKernelConfig& config) const {
    std::shared_ptr<BrgemmAMXCompiledKernel> compiled_kernel = std::make_shared<BrgemmAMXCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty()) {
        return compiled_kernel;
    }

    const auto& cache = m_kernel_cache.lock();
    OPENVINO_ASSERT(cache, "Invalid kernel cache pointer in BrgemmAMXKernelExecutor::compile_kernel()");

    auto brgemm_key = [&config](int64_t K, int64_t LDA, float beta) {
        auto key = config;
        key.update(config.get_M(), config.get_N(), K, LDA, config.get_LDB(), config.get_LDC(), beta);
        return key;
    };

    auto brgemm_builder = [](const BrgemmAMXKernelConfig& k) {
        std::shared_ptr<BrgemmAMXCompiledKernel::BrgemmKernel> ker =
            std::make_shared<BrgemmAMXCompiledKernel::BrgemmKernel>();
        create_brgemm_kernel(ker->brgemm_kernel,
                             k.get_dt_in0(),
                             k.get_dt_in1(),
                             k.get_isa(),
                             k.get_M(),
                             k.get_N(),
                             k.get_K(),
                             k.get_LDA(),
                             k.get_LDB(),
                             k.get_LDC(),
                             k.get_beta(),
                             true,
                             ker->palette);
        return ker;
    };

    auto brgemm_copy_a_builder = [](const BrgemmCopyAKey& k) {
        std::shared_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> ker{nullptr};
        create_brgemm_copy_a_kernel(ker, k.isa, k.dt, k.K, k.K_blk, k.K_tail, k.src_stride, k.LDA);
        return ker;
    };

    auto K_tail = config.get_K() % config.get_inner_K_blk();
    auto K_body = config.get_K() - K_tail;

    float beta = config.get_beta();

    // Brgemm Kernel for K_body
    if (K_body != 0) {
        const auto result = cache->getOrCreate(brgemm_key(K_body, config.get_LDA(), beta), brgemm_builder);
        compiled_kernel->K_body_kernel = result.first;
        beta = 1;
    }

    // Brgemm Kernel for K_tail with BrgemmCopyA if needed
    if (K_tail != 0) {
        auto LDA = config.get_LDA();
        if (config.need_copy_a(K_tail)) {
            const auto copy_A_src_stride = LDA * dnnl_data_type_size(config.get_dt_in0());
            K_tail = ov::snippets::utils::rnd_up(K_tail, config.get_vnni_factor());
            LDA = K_tail;

            const auto key = BrgemmCopyAKey(config.get_isa(),
                                            config.get_dt_in0(),
                                            config.get_K(),
                                            config.get_inner_K_blk(),
                                            K_tail,
                                            copy_A_src_stride,
                                            LDA);
            const auto result = cache->getOrCreate(key, brgemm_copy_a_builder);
            compiled_kernel->brgemm_copy_a_kernel = result.first;
        }

        const auto result = cache->getOrCreate(brgemm_key(K_tail, LDA, beta), brgemm_builder);
        compiled_kernel->K_tail_kernel = result.first;
    }

    return compiled_kernel;
}

void BrgemmAMXKernelExecutor::create_brgemm_copy_a_kernel(
    std::shared_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& kernel,
    dnnl::impl::cpu::x64::cpu_isa_t isa,
    dnnl_data_type_t dt,
    dnnl_dim_t K,
    dnnl_dim_t K_blk,
    dnnl_dim_t K_tail,
    dnnl_dim_t src_stride,
    dnnl_dim_t LDA) {
    matmul::brgemm_matmul_conf_t conf_;
    conf_.src_tag = dnnl_abcd;  // unused
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

    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> brgemm_matmul_copy_a = nullptr;
    OV_CPU_JIT_EMITTER_ASSERT(create_brgemm_matmul_copy_a(brgemm_matmul_copy_a, &conf_) == dnnl_success,
                              "Cannot create brgemm copy a kernel due to invalid params");
    kernel = std::move(brgemm_matmul_copy_a);
}

void BrgemmAMXKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                            const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                            BrgemmAMXKernelConfig& config) const {
    return BrgemmBaseKernelExecutor::update_config(expr, linear_ir, config);
}

void BrgemmAMXKernelExecutor::configure_tiles_if_needed(amx_tile_config_t* config,
                                                        const char* palette,
                                                        dnnl_dim_t M,
                                                        dnnl_dim_t N,
                                                        dnnl_dim_t K) {
    auto compatible = [&](amx_tile_config_t* rhs) {
        return rhs && rhs->M == M && rhs->N == N && rhs->K == K;
    };
    if (config && !compatible(config)) {
        config->M = M;
        config->N = N;
        config->K = K;
        cpu::x64::amx_tile_configure(palette);
    }
}

void BrgemmAMXKernelExecutor::execute_brgemm_copy_a_kernel(
    const std::shared_ptr<matmul::jit_brgemm_matmul_copy_a_t>& kernel,
    const void* src,
    const void* tr_src,
    dnnl_dim_t M,
    dnnl_dim_t K) {
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
        const auto& K_body_kernel = kernel->K_body_kernel;
        configure_tiles_if_needed(args->amx_tile_config,
                                  K_body_kernel->palette,
                                  config.get_M(),
                                  config.get_N(),
                                  K_body);
        execute_brgemm_kernel(K_body_kernel->brgemm_kernel, src_ptr, wei_ptr, args->C, scratch, false);

        src_ptr = src_ptr + K_body * dnnl_data_type_size(config.get_dt_in0());
        wei_ptr = wei_ptr + (K_body * config.get_LDB()) * dnnl_data_type_size(config.get_dt_in1());
    }

    if (K_tail != 0) {
        if (config.need_copy_a(K_tail)) {
            auto* tr_src = scratch + BrgemmCPU::SCRATCH_BYTE_SIZE;

            execute_brgemm_copy_a_kernel(kernel->brgemm_copy_a_kernel, src_ptr, tr_src, config.get_M(), K_tail);
            src_ptr = tr_src;
        }

        const auto& K_tail_kernel = kernel->K_tail_kernel;
        configure_tiles_if_needed(args->amx_tile_config,
                                  K_tail_kernel->palette,
                                  config.get_M(),
                                  config.get_N(),
                                  K_tail);
        execute_brgemm_kernel(K_tail_kernel->brgemm_kernel, src_ptr, wei_ptr, args->C, scratch, false);
    }
}

#undef INNER_K_BLK
#undef VNNI_FACTOR
#undef EQ
#undef HASH

}  // namespace ov::intel_cpu::x64
