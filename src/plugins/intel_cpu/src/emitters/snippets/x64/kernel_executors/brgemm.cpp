// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "snippets/lowered/loop_manager.hpp"

#include <cpu/x64/amx_tile_configure.hpp>
#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)
#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace {
size_t init_hash(dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1, float beta, bool is_with_amx,
                 bool is_with_comp, dnnl::impl::cpu::x64::cpu_isa_t isa) {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(dt_in0); HASH(dt_in1);
    HASH(is_with_amx); HASH(is_with_comp);
    HASH(beta); HASH(isa);
#undef HASH
    return seed;
}
} // namespace

namespace ov {
namespace intel_cpu {
BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                       float beta, bool is_with_amx, bool is_with_comp,
                                       dnnl::impl::cpu::x64::cpu_isa_t primitive_isa) :
                                       m_static_params(std::make_shared<StaticParams>(in0_dtype, in1_dtype, beta,
                                                                                      is_with_amx, is_with_comp,
                                                                                      primitive_isa)) {
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_completed() const {
    return !utils::one_of(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC) || is_empty();
}

bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(m_hash) &&
           EQ(m_M) && EQ(m_N) && EQ(m_K) &&
           EQ(m_LDA) && EQ(m_LDB) && EQ(m_LDC) &&
           (EQ(m_static_params.get()) || *m_static_params == *(rhs.m_static_params));
#undef EQ
}

void BrgemmKernelConfig::update(dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, dnnl_dim_t LDA, dnnl_dim_t LDB, dnnl_dim_t LDC) {
    // If M is zero, it means that Brgemm won't be executed (in Loop with work_amount = 0, for example)
    // To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (utils::one_of(0, M, N, K)) {
        m_M = 0; m_N = 0; m_K = 0;
        m_LDA = 0; m_LDB = 0; m_LDC = 0;
    } else {
        m_M = M; m_N = N; m_K = K;
        m_LDA = LDA; m_LDB = LDB; m_LDC = LDC;
    }
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_empty() const {
    return everyone_is(0, m_M, m_N, m_K, m_LDA, m_LDB, m_LDC);
}

BrgemmKernelConfig::operator amx_tile_config_t() const {
    amx_tile_config_t res;
    res.M = m_M; res.N = m_N; res.K = m_K;
    return  res;
}

BrgemmKernelConfig::StaticParams::StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype,
                                               float beta, bool is_with_amx, bool is_with_comp,
                                               dnnl::impl::cpu::x64::cpu_isa_t primitive_isa) :
                                               dt_in0(DTYPE_CAST(in0_dtype)), dt_in1(DTYPE_CAST(in1_dtype)),
                                               beta(beta), is_with_amx(is_with_amx), is_with_comp(is_with_comp),
                                               isa(primitive_isa),
                                               hash(init_hash(dt_in0, dt_in1, beta, is_with_amx, is_with_comp, isa)) {
}

bool BrgemmKernelConfig::StaticParams::operator==(const StaticParams& rhs) const {
#define EQ(X) X == rhs.X
    return EQ(hash) &&
           EQ(dt_in0) && EQ(dt_in1) && EQ(beta) &&
           EQ(is_with_amx) && EQ(is_with_comp) && EQ(isa);
#undef EQ
}
size_t BrgemmKernelConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(m_M); HASH(m_N); HASH(m_K);
    HASH(m_LDA); HASH(m_LDB); HASH(m_LDC);
#undef HASH
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#define PRINT(X) ss << #X  << " = " << X << "\n"
std::string BrgemmKernelConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(dt_in0); PRINT(dt_in1);
    PRINT(is_with_amx); PRINT(is_with_comp);
    PRINT(beta); PRINT(isa);
    return ss.str();
}

std::string BrgemmKernelConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_M); PRINT(m_N); PRINT(m_K);
    PRINT(m_LDA); PRINT(m_LDB); PRINT(m_LDC);
    return ss.str();
}
#undef PRINT
#endif

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config) :
        CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), std::move(config)) { }


std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const BrgemmKernelConfig& config) const {
    std::shared_ptr<BrgemmCompiledKernel> compiled_kernel = std::make_shared<BrgemmCompiledKernel>();

    // Brgemm is not executable - nothing to compile
    if (config.is_empty())
        return compiled_kernel;

    cpu::x64::brgemm_t desc;
    auto status = brgemm_desc_init(&desc, config.get_isa(), cpu::x64::brgemm_strd,
                                   config.get_dt_in0(), config.get_dt_in1(),
                                   false, false, cpu::x64::brgemm_row_major, 1.f,
                                   config.get_beta(),
                                   config.get_LDA(), config.get_LDB(), config.get_LDC(),
                                   config.get_M(), config.get_N(), config.get_K(), nullptr);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm descriptor due to invalid params");

    if (config.is_with_amx()) {
        status = brgemm_init_tiles(desc, compiled_kernel->palette);
        OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm tiles due to invalid params");
    }

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    status = brgemm_kernel_create(&kernel_, desc);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot create brgemm kernel due to invalid params");
    compiled_kernel->compiled_kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);

    return compiled_kernel;
}
void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                         const ov::snippets::lowered::LinearIRPtr& linear_ir,
                                         BrgemmKernelConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    const auto in1_shape = snippets::utils::get_planar_vdims(input_pds[1]->get_shape(), input_pds[1]->get_layout());
    auto in0_subtensor = input_pds[0]->get_subtensor();
    auto in1_subtensor = input_pds[1]->get_subtensor();

    auto M = *++in0_subtensor.rbegin();
    auto K = *in0_subtensor.rbegin();
    auto N = *in1_subtensor.rbegin();

    if (ov::snippets::utils::is_full_dim_value(M)) {
        M = *++in0_shape.rbegin();
    } else {
        const auto& loop_ids = expr->get_loop_ids();
        OPENVINO_ASSERT(!loop_ids.empty(), "Loop by dimension M is missed");
        // TODO [146125]: Loop by M is first one in `loop_ids`
        const auto& expanded_loop_info = linear_ir->get_loop_manager()->get_loop_info<ov::snippets::lowered::ExpandedLoopInfo>(loop_ids.front());
        M = expanded_loop_info->get_increment();
        input_pds[0]->set_subtensor_dim(1, M);
        output_pds[0]->set_subtensor_dim(1, M);
    }

    if (ov::snippets::utils::is_full_dim_value(K)) {
        K = *in0_shape.rbegin();
    } else if (ov::snippets::utils::is_dynamic_value(K)) {
        OPENVINO_THROW("Dynamic K is not supported");
    }

    if (ov::snippets::utils::is_full_dim_value(N)) {
        N = *in1_shape.rbegin();
    } else if (ov::snippets::utils::is_dynamic_value(N)) {
        OPENVINO_THROW("Dynamic N is not supported");
    }

    const auto LDA = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(0)));
    const auto LDC = DIM_CAST(snippets::utils::get_dim_stride(expr->get_output_port(0)));
    auto LDB = DIM_CAST(snippets::utils::get_dim_stride(expr->get_input_port(1)));

    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_node, "Got invalid node type in update_config");
    if (with_repacking(brgemm_node->get_type())) {
        const auto repacking_buffer_shape = brgemm_node->get_brgemm_copy()->get_repacking_buffer_shape();
        OV_CPU_JIT_EMITTER_ASSERT(!repacking_buffer_shape.empty(), "Repacking buffer shape mustn't be empty");
        LDB = DIM_CAST(repacking_buffer_shape.back());
    }

    config.update(DIM_CAST(M), DIM_CAST(N), DIM_CAST(K), LDA, LDB, LDC);
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* executor, call_args* args) {
    auto kernel = executor->get_kernel();
    const auto& config = static_cast<const BrgemmKernelConfig&>(executor->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel, "has nullptr compiler kernel or invalid config");

    const auto tile_config = args->amx_tile_config;
    if (config.is_with_amx() && tile_config && !config.compatible(tile_config)) {
        *tile_config = static_cast<amx_tile_config_t>(config);
        cpu::x64::amx_tile_configure(kernel->palette);
    }

    cpu::x64::brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = args->A;
    brgemm_p.ptr_B = args->B;
    brgemm_p.ptr_C = args->C;
    brgemm_p.ptr_D = args->C;
    brgemm_p.ptr_buf = args->scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(config.is_with_comp());
    brgemm_p.do_apply_comp = static_cast<size_t>(config.is_with_comp());
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(kernel->compiled_kernel, "has nullptr kernel");
    (*kernel->compiled_kernel)(&brgemm_p);
}

#ifdef SNIPPETS_DEBUG_CAPS
BrgemmKernelReferenceExecutor::BrgemmKernelReferenceExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config) :
        BrgemmKernelExecutor(std::move(kernel_cache), std::move(config)) {
}

std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelReferenceExecutor::compile_kernel(const BrgemmKernelConfig& c) const {
    const auto& res = std::make_shared<BrgemmCompiledKernel>();
    res->compiled_kernel.reset(new brgemm_ref_kernel(c));
    return res;
}

brgemm_ref_kernel::brgemm_ref_kernel(BrgemmKernelConfig c) : m_config(std::move(c)) {
    OV_CPU_JIT_EMITTER_ASSERT(!m_config.is_with_comp() && !m_config.is_with_amx(),
                              "brgemm_ref_kernel doesn't currently support compensations or amx");
    OV_CPU_JIT_EMITTER_ASSERT(m_config.get_dt_in0() == m_config.get_dt_in1() &&
                              m_config.get_dt_in0() == dnnl_data_type_t::dnnl_f32,
                              "brgemm_ref_kernel currently supports only fp32 inputs");
}

void brgemm_ref_kernel::operator()(dnnl::impl::cpu::x64::brgemm_kernel_params_t* args) const {
    auto A = reinterpret_cast<const float*>(args->ptr_A);
    auto B = reinterpret_cast<const float*>(args->ptr_B);
    auto C = reinterpret_cast<float*>(args->ptr_C);
    for (dnnl_dim_t m = 0; m < m_config.get_M(); m++) {
        for (dnnl_dim_t n = 0; n < m_config.get_N(); n++, B++) {
            C[n] = 0;
            for (dnnl_dim_t k = 0; k < m_config.get_K(); k++)
                C[n] += A[k] * B[k * m_config.get_LDB()];
        }
        B -= m_config.get_N();
        A += m_config.get_LDA();
        C += m_config.get_LDC();
    }
}
#endif

}   // namespace intel_cpu
}   // namespace ov
