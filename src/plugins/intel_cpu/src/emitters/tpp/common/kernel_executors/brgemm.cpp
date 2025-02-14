// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "emitters/tpp/common/utils.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

#define PRINT(X) ss << #X << " = " << X << "\n"
#define HASH(X)  seed = dnnl::impl::hash_combine(seed, X)

namespace ov::intel_cpu::tpp {

BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype)
    : BrgemmGenericKernelConfig(),
      m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype)) {}

bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) &&
           (get_static_params() == rhs.get_static_params() || *get_static_params() == *(rhs.get_static_params()));
}

size_t BrgemmKernelConfig::compute_hash() const {
    size_t static_seed = get_static_params()->hash();
    size_t dynamic_seed = BrgemmGenericKernelConfig::compute_hash();
    return dnnl::impl::hash_combine(static_seed, dynamic_seed);
}

void BrgemmKernelConfig::update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    // update compile flag, which should be reset depend on beta. It is combination of beta and static_compile_flag and
    // considered in hash() and operator==
    libxsmm_bitfield new_flag = get_static_compile_flags();
    if (beta == 0) {
        new_flag |= LIBXSMM_GEMM_FLAG_BETA_0;
    }
    set_compile_flags(new_flag);
    m_hash = compute_hash();
}

BrgemmKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype) {
    m_type_in0 = tpp::utils::ov_to_xsmm_dtype(in0_dtype);
    m_type_in1 = tpp::utils::ov_to_xsmm_dtype(in1_dtype);
    m_type_exec = LIBXSMM_DATATYPE_F32;
    m_type_out0 = LIBXSMM_DATATYPE_F32;
    m_compile_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    m_prefetching_flags = false;
    m_hash = compute_hash();
}

size_t BrgemmKernelConfig::StaticParams::compute_hash() {
    size_t seed = 0;
    HASH(m_type_in0);
    HASH(m_type_in1);
    HASH(m_type_exec);
    HASH(m_type_out0);
    HASH(m_compile_flags);
    HASH(m_prefetching_flags);
    return seed;
}

bool BrgemmKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return m_type_in0 == rhs.m_type_in0 && m_type_in1 == rhs.m_type_in1 && m_type_exec == rhs.m_type_exec &&
           m_type_out0 == rhs.m_type_out0 && m_compile_flags == rhs.m_compile_flags &&
           m_prefetching_flags == rhs.m_prefetching_flags;
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(m_type_in0);
    PRINT(m_type_in1);
    PRINT(m_type_out0);
    PRINT(m_type_exec);
    PRINT(m_compile_flags);
    PRINT(m_prefetching_flags);
    return ss.str();
}

std::string BrgemmKernelConfig::to_string() const {
    std::stringstream ss;
    ss << get_static_params()->to_string() << "\n";
    ss << BrgemmGenericKernelConfig::to_string() << "\n";
    PRINT(m_compile_flags);
    return ss.str();
}
#endif

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config)
    : CPUKernelExecutor<BrgemmKernelConfig, BrgemmTppCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmTppCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    std::shared_ptr<BrgemmTppCompiledKernel> compiled_kernel = std::make_shared<BrgemmTppCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty())
        return compiled_kernel;
    // data is row major, but libxsmm gemm suppose column major. in0 and in1 are exchanged to avoid data repack(kernel
    // call args aligned).
    libxsmm_gemm_shape m_shape = libxsmm_create_gemm_shape(config.get_N(),
                                                           config.get_M(),
                                                           config.get_K(),
                                                           config.get_LDB(),
                                                           config.get_LDA(),
                                                           config.get_LDC(),
                                                           config.get_type_in1(),
                                                           config.get_type_in0(),
                                                           config.get_type_out0(),
                                                           config.get_type_exec());
    compiled_kernel->brgemm_kernel =
        std::make_shared<libxsmm_gemmfunction>(reinterpret_cast<const libxsmm_gemmfunction>(COMPILE_TPP_KERNEL(
            libxsmm_dispatch_gemm(m_shape, config.get_compile_flags(), config.get_prefetching_flags()))));

    return compiled_kernel;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    int64_t M, N, K, beta;
    std::tie(M, N, K, beta) = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);
    const auto& tpp_mod = std::dynamic_pointer_cast<tpp::modifier::TensorProcessingPrimitive>(expr->get_node());
    auto replace_full_dim = [](size_t dim, size_t replace_dim) {
        if (ov::snippets::utils::is_full_dim_value(dim))
            return replace_dim;
        return dim;
    };

    const auto num_ins = expr->get_node()->get_input_size();
    const auto num_outs = expr->get_node()->get_output_size();

    std::vector<size_t> io_strides(num_ins + num_outs);

    for (size_t i = 0; i < num_ins; i++) {
        io_strides[i] =
            replace_full_dim(tpp_mod->get_input_stride(i), expr->get_input_port_descriptor(i)->get_shape().back());
    }

    for (size_t i = 0; i < num_outs; i++) {
        const auto i_off = i + num_ins;
        io_strides[i_off] =
            replace_full_dim(tpp_mod->get_output_stride(i), expr->get_output_port_descriptor(i)->get_shape().back());
    }

    config.update(M, N, K, io_strides[0], io_strides[1], io_strides[2], beta);
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    libxsmm_gemm_param gemm_p;
    // data is row major, but libxsmm gemm suppose column major. in0 and in1 are exchanged to avoid data repack(kernel
    // creation params aligned).
    gemm_p.a.primary = in1;
    gemm_p.b.primary = in0;
    gemm_p.c.primary = out0;
    auto brg_kernel = executor->get_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel, "has nullptr compiler kernel");
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel->brgemm_kernel, "has nullptr compiler brgemm_kernel");
    (*(brg_kernel->brgemm_kernel))(&gemm_p);
}

#undef PRINT
#undef HASH

}  // namespace ov::intel_cpu::tpp
