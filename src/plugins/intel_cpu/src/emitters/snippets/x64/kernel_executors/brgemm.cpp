// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include <cpu/x64/amx_tile_configure.hpp>
#include "common/utils.hpp"
#include "dnnl_extension_utils.h"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#define DIM_CAST(X) static_cast<dnnl_dim_t>(X)
#define DTYPE_CAST(X) static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(X))

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
BrgemmKernelConfig::BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype, float beta,
                                       bool is_with_amx, bool is_with_comp,
                                       size_t M, size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC) :
                     dt_in0(DTYPE_CAST(in0_dtype)), dt_in1(DTYPE_CAST(in1_dtype)),
                     is_with_amx(is_with_amx), is_with_comp(is_with_comp), beta(beta),
                     M(DIM_CAST(M)), N(DIM_CAST(N)), K(DIM_CAST(K)),
                     LDA(DIM_CAST(LDA)), LDB(DIM_CAST(LDB)), LDC(DIM_CAST(LDC)) {
    bool is_int8 = utils::one_of(dt_in0, data_type::u8, data_type::s8) &&
                   utils::one_of(dt_in1, data_type::u8, data_type::s8);
    isa = is_with_amx ?
            cpu::x64::avx512_core_amx :
            dt_in0 == dnnl_data_type_t::dnnl_bf16 ?
                cpu::x64::avx512_core_bf16 :
                is_int8 ?
                    cpu::x64::avx512_core_vnni :
                    cpu::x64::avx512_core;
    // Note: hash could be initialized only after ias
    m_hash = compute_hash();
}

bool BrgemmKernelConfig::is_completed() const {
    return !utils::one_of(0, M, N, K, LDA, LDB, LDC);
}

size_t BrgemmKernelConfig::compute_hash() const {
    size_t seed = 0;
#define HASH(X) seed = hash_combine(seed, X)
    HASH(dt_in0); HASH(dt_in1);
    HASH(is_with_amx); HASH(is_with_comp);
    HASH(beta); HASH(isa);
    HASH(M); HASH(N); HASH(K);
    HASH(LDA); HASH(LDB); HASH(LDC);
#undef HASH
    return seed;
}
bool BrgemmKernelConfig::operator==(const BrgemmKernelConfig& rhs) const {
#define EQUAL(X) X == rhs.X
    return EQUAL(dt_in0) && EQUAL(dt_in1) &&
           EQUAL(is_with_amx)  && EQUAL(is_with_comp) &&
           EQUAL(beta) && EQUAL(isa) &&
           EQUAL(M) && EQUAL(N) && EQUAL(K) &&
           EQUAL(LDA) && EQUAL(LDB) && EQUAL(LDC) && EQUAL(m_hash);
#undef EQUAL
}
bool BrgemmKernelConfig::operator!=(const BrgemmKernelConfig& rhs) const {
    return !(*this == rhs);
}

#ifdef SNIPPETS_DEBUG_CAPS
std::string BrgemmKernelConfig::to_string() const {
    std::stringstream ss;
#define PRINT(X) ss << #X  << " = " << X << "\n"
    PRINT(dt_in0); PRINT(dt_in1);
    PRINT(is_with_amx); PRINT(is_with_comp);
    PRINT(beta); PRINT(isa);
    PRINT(M); PRINT(N); PRINT(K);
    PRINT(LDA); PRINT(LDB); PRINT(LDC);
#undef PRINT
    return ss.str();
}
#endif

BrgemmKernelExecutor::BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache,
                                           const std::shared_ptr<BrgemmKernelConfig>& config) :
        CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel>(std::move(kernel_cache), config) { }


std::shared_ptr<BrgemmCompiledKernel> BrgemmKernelExecutor::compile_kernel(const std::shared_ptr<const BrgemmKernelConfig>& config) const {
    OV_CPU_JIT_EMITTER_ASSERT(config, "Invalid config provided for BrgemmKernelDesc::compile_kernel");
    cpu::x64::brgemm_t desc;
    auto status = brgemm_desc_init(&desc, config->isa, cpu::x64::brgemm_strd,
                                   config->dt_in0, config->dt_in1,
                                   false, false, cpu::x64::brgemm_row_major, 1.f,
                                   config->beta,
                                   config->LDA, config->LDB, config->LDC,
                                   config->M, config->N, config->K, nullptr);

    std::shared_ptr<BrgemmCompiledKernel> compiled_kernel = std::make_shared<BrgemmCompiledKernel>();

    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm descriptor due to invalid params");
    if (config->is_with_amx) {
        status = brgemm_init_tiles(desc, compiled_kernel->palette);
        OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot initialize brgemm tiles due to invalid params");
    }

    cpu::x64::brgemm_kernel_t* kernel_ = nullptr;
    status = brgemm_kernel_create(&kernel_, desc);
    OV_CPU_JIT_EMITTER_ASSERT(status == dnnl_success, "Cannot create brgemm kernel due to invalid params");
    compiled_kernel->compiled_kernel = std::unique_ptr<brgemm_kernel_t>(kernel_);

    return compiled_kernel;
}

void BrgemmKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr, std::shared_ptr<BrgemmKernelConfig>& config) const {
    auto get_projected_input_subtensor = [](const snippets::lowered::PortDescriptorPtr& desc) {
        // Note: for output shape you will need get_preordered_vdims()
        auto shape = snippets::utils::get_planar_vdims(desc->get_shape(), desc->get_layout());
        auto subtensor = desc->get_subtensor();
        OV_CPU_JIT_EMITTER_ASSERT(subtensor.size() <= shape.size() && subtensor.size() == 2,
                                  "Invalid subtensor + shape combination");
        auto shape_it = shape.rbegin();
        for (auto sub_it = subtensor.rbegin(); sub_it != subtensor.rend(); sub_it++, shape_it++) {
            *sub_it = std::min(*sub_it, *shape_it);
        }
        return subtensor;
    };
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 2 || input_pds.size() == 3) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");
    // Update runtime-defined config fields:
    // Matrix A (first input)
    config->LDA = DIM_CAST(snippets::utils::get_in_leading_dim(input_pds[0]));
    const auto& in0_subtensor = get_projected_input_subtensor(input_pds[0]);
    config->K = DIM_CAST(*in0_subtensor.rbegin());
    config->M = DIM_CAST(*++in0_subtensor.rbegin());
    // Matrix B (second input)
    // Non float input 1 => with data repacking
    if (config->dt_in1 != dnnl_f32) {
        const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
        const auto repacking_buffer_shape = brgemm_node->get_brgemm_copy()->get_repacking_buffer_shape();
        OV_CPU_JIT_EMITTER_ASSERT(!repacking_buffer_shape.empty(), "Repacking buffer shape mustn't be empty");
        config->LDB = DIM_CAST(repacking_buffer_shape.back());
    } else {
        config->LDB = DIM_CAST(snippets::utils::get_in_leading_dim(input_pds[1]));
    }
    config->N = DIM_CAST(*get_projected_input_subtensor(input_pds[1]).rbegin());
    // Matrix C (output)
    config->LDC = DIM_CAST(snippets::utils::get_out_leading_dim(output_pds[0]));
    config->m_hash = config->compute_hash();
}

void BrgemmKernelExecutor::execute(const BrgemmKernelExecutor* desc, call_args* args) {
    const auto& kernel = desc->get_kernel();
    const auto& config = std::static_pointer_cast<const BrgemmKernelConfig>(desc->get_config());
    OV_CPU_JIT_EMITTER_ASSERT(kernel && config, "has nullptr compiler kernel or invalid config");

    if (config->is_with_amx) {
        const auto& amx_tile_config = args->amx_tile_config;
        if (config->M != amx_tile_config->M || config->K != amx_tile_config->K || config->N != amx_tile_config->N) {
            amx_tile_config->M = config->M;
            amx_tile_config->K = config->K;
            amx_tile_config->N = config->N;
            cpu::x64::amx_tile_configure(kernel->palette);
        }
    }

    cpu::x64::brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = nullptr;  // default value
    brgemm_p.ptr_A = args->A;
    brgemm_p.ptr_B = args->B;
    brgemm_p.ptr_C = args->C;
    brgemm_p.ptr_D = args->C;
    brgemm_p.ptr_buf = args->scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = static_cast<size_t>(config->is_with_comp);
    brgemm_p.do_apply_comp = static_cast<size_t>(config->is_with_comp);
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = 1;  // default value
    OV_CPU_JIT_EMITTER_ASSERT(kernel->compiled_kernel, "has nullptr kernel");
    (*kernel->compiled_kernel)(&brgemm_p);
}

}   // namespace intel_cpu
}   // namespace ov
