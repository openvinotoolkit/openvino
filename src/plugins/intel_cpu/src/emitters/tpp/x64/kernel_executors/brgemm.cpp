// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
#define COMPILE_BRGEMM_TPP_KERNEL(...)                                        \
    [&]() {                                                                   \
        setenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX", "1", 1);      \
        setenv("LIBXSMM_GEMM_K_A_PF_DIST", "4", 1);                           \
        auto res = reinterpret_cast<const libxsmm_gemmfunction>(__VA_ARGS__); \
        unsetenv("LIBXSMM_X86_HINT_USE_HIGH_PREC_ELTWISE_APPROX");            \
        unsetenv("LIBXSMM_GEMM_K_A_PF_DIST");                                 \
        return res;                                                           \
    }()

BrgemmTppKernelConfig::BrgemmTppKernelConfig(const element::Type& in0_dtype,
                                             const element::Type& in1_dtype,
                                             libxsmm_bitfield compile_flags,
                                             bool prefetching_flags,
                                             dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : BrgemmBaseKernelConfig(),
      m_static_params(
          std::make_shared<StaticParams>(in0_dtype, in1_dtype, prefetching_flags, primitive_isa)) {
    m_hash = compute_hash();
}

BrgemmTppKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype,
                                                  const element::Type& in1_dtype,
                                                  bool prefetching_flags,
                                                  dnnl::impl::cpu::x64::cpu_isa_t primitive_isa)
    : StaticBaseParams(in0_dtype, in1_dtype, primitive_isa, compute_hash(prefetching_flags)),
      m_prefetching_flags(prefetching_flags) {
        m_type_in0 = tpp::utils::ov_to_xsmm_dtype(in0_dtype);
        m_type_in1 = tpp::utils::ov_to_xsmm_dtype(in1_dtype);
        m_type_exec = m_type_in0 == LIBXSMM_DATATYPE_I8 || m_type_in0 == LIBXSMM_DATATYPE_U8 ?
            LIBXSMM_DATATYPE_I32 :
            LIBXSMM_DATATYPE_F32;
        m_type_out0 = m_type_exec == LIBXSMM_DATATYPE_I32 ?
            LIBXSMM_DATATYPE_I32 :
            LIBXSMM_DATATYPE_F32;
        const bool is_f32_gemm = m_type_in0 == m_type_in1 && m_type_in0 == LIBXSMM_DATATYPE_F32;
        const bool is_bf16_gemm =  m_type_in0 == m_type_in1 && m_type_in0 == LIBXSMM_DATATYPE_BF16;
        const bool is_i8_gemm = m_type_in0 == LIBXSMM_DATATYPE_U8 || m_type_in0 == LIBXSMM_DATATYPE_I8;

        m_compile_flags = is_f32_gemm ?
                      LIBXSMM_GEMM_FLAGS('N', 'N') :
                      LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') |
                      LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG |
                      LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
        if (m_type_in0 == LIBXSMM_DATATYPE_U8) {
            m_type_in0 = LIBXSMM_DATATYPE_I8;
            m_compile_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
       }
        if (m_type_in1 == LIBXSMM_DATATYPE_U8) {
            m_type_in1 = LIBXSMM_DATATYPE_I8;
            m_compile_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
        }
      }

bool BrgemmTppKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return StaticBaseParams::operator==(rhs) && m_prefetching_flags == rhs.m_prefetching_flags;
}

size_t BrgemmTppKernelConfig::StaticParams::compute_hash(bool prefetching_flags) {
    return hash_combine(0, prefetching_flags);
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmTppKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    ss << StaticBaseParams::to_string();
    ss << "m_prefetching_flags = " << m_prefetching_flags << "\n";
    return ss.str();
}
#endif

BrgemmTppKernelExecutor::BrgemmTppKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                                                 BrgemmTppKernelConfig config)
    : CPUKernelExecutor<BrgemmTppKernelConfig, BrgemmTppCompiledKernel>(std::move(kernel_cache), std::move(config)) {}

std::shared_ptr<BrgemmTppCompiledKernel> BrgemmTppKernelExecutor::compile_kernel(
    const BrgemmTppKernelConfig& config) const {
    std::shared_ptr<BrgemmTppCompiledKernel> compiled_kernel = std::make_shared<BrgemmTppCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty())
        return compiled_kernel;

    libxsmm_gemm_shape m_shape = libxsmm_create_gemm_shape(config.get_N(),
                                                           config.get_M(),
                                                           config.get_K(),
                                                           config.get_LDB(),
                                                           config.get_LDA(),
                                                           config.get_LDC(),
                                                           config.get_type_in0(),
                                                           config.get_type_in1(),
                                                           config.get_type_out0(),
                                                           config.get_type_exec());
    const auto& compile_flag = config.get_compile_flags();
    auto refreshed_compile_flag = config.get_beta() == 0 ? config.get_compile_flags() | LIBXSMM_GEMM_FLAG_BETA_0 :
                                                        compile_flag;
    compiled_kernel->brgemm_kernel = std::make_shared<libxsmm_gemmfunction>(COMPILE_BRGEMM_TPP_KERNEL(
        libxsmm_dispatch_gemm(m_shape, refreshed_compile_flag, config.get_prefetching_flags())));

    return compiled_kernel;
}

void BrgemmTppKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                            const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                            BrgemmTppKernelConfig& config) const {
    BrgemmBaseKernelExecutor::update_config(expr, linear_ir, config);
    const auto& tpp_mod = std::dynamic_pointer_cast<tpp::modifier::TensorProcessingPrimitive>(expr->get_node());

    auto replace_full_dim = [](size_t dim, size_t replace_dim) {
        if (ov::snippets::utils::is_full_dim_value(dim))
            return replace_dim;
        return dim;
    };

    const auto num_ins = expr->get_node()->get_input_size();
    const auto num_outs = expr->get_node()->get_output_size();

    size_t io_strides[num_ins + num_outs];

    for (size_t i = 0; i < num_ins; i++) {
        io_strides[i] =
            replace_full_dim(tpp_mod->get_input_stride(i), expr->get_input_port_descriptor(i)->get_shape().back());
    }

    for (size_t i = 0; i < num_outs; i++) {
        const auto i_off = i + num_ins;
        io_strides[i_off] =
            replace_full_dim(tpp_mod->get_output_stride(i), expr->get_output_port_descriptor(i)->get_shape().back());
    }

    config.update(config.get_M(), config.get_N(), config.get_K(), io_strides[0], io_strides[1], io_strides[2], config.get_beta());
}

void BrgemmTppKernelExecutor::execute(const BrgemmTppKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    libxsmm_gemm_param gemm_p;
    gemm_p.a.primary = in1;
    gemm_p.b.primary = in0;
    gemm_p.c.primary = out0;
    auto brg_kernel = executor->get_kernel();
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel, "has nullptr compiler kernel");
    OV_CPU_JIT_EMITTER_ASSERT(brg_kernel->brgemm_kernel, "has nullptr compiler brgemm_kernel");
    (*(brg_kernel->brgemm_kernel))(&gemm_p);
}

}  // namespace intel_cpu
}  // namespace ov
