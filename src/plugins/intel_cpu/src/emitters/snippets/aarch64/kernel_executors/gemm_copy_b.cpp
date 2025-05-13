// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gemm_copy_b.hpp"

// #include "emitters/tpp/common/utils.hpp"
#include "openvino/core/parallel.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu::aarch64 {

GemmCopyBKernelKaiConfig::GemmCopyBKernelKaiConfig(const size_t N, const size_t K, const size_t n_blk_size)
    : m_N(N),
      m_K(K),
      m_n_blk_size(n_blk_size),
      m_hash(compute_hash()) {}

bool GemmCopyBKernelKaiConfig::is_completed() const {
    return !ov::snippets::utils::one_of(0, m_N, m_K, m_n_blk_size);
}
#define PRINT(X) ss << #X << " = " << (X) << "\n"
#ifdef SNIPPETS_DEBUG_CAPS
std::string GemmCopyBKernelKaiConfig::to_string() const {
    std::stringstream ss;
    PRINT(m_N);
    PRINT(m_K);
    PRINT(m_n_blk_size);
    return ss.str();
}
#endif

void GemmCopyBKernelKaiConfig::update(size_t N, size_t K, size_t n) {
    // If one of the dims is zero, it means that BrgemmCopyB won't be executed (in Loop with work_amount = 0, for
    // example) To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (ov::snippets::utils::one_of(0, N, K, n)) {
        m_N = 0;
        m_K = 0;
        m_n_blk_size = 0;
    } else {
        m_N = N;
        m_K = K;
        m_n_blk_size = n;
    }
    m_hash = compute_hash();
}

size_t GemmCopyBKernelKaiConfig::compute_hash() const {
    size_t seed = 0;
    seed = dnnl::impl::hash_combine(seed, m_N);
    seed = dnnl::impl::hash_combine(seed, m_K);
    seed = dnnl::impl::hash_combine(seed, m_n_blk_size);
    return seed;
}

GemmCopyBKaiKernelExecutor::GemmCopyBKaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBKaiCompiledKernel>(std::move(config)) {}

void GemmCopyBKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                               const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                               GemmCopyBKernelKaiConfig& config) const {
    const auto& input_pds = expr->get_input_port_descriptors();
    const auto& output_pds = expr->get_output_port_descriptors();
    OV_CPU_JIT_EMITTER_ASSERT((input_pds.size() == 1) && output_pds.size() == 1,
                              "Invalid number of in/out port descriptors");

    const auto& in0_shape = snippets::utils::get_planar_vdims(input_pds[0]->get_shape(), input_pds[0]->get_layout());
    int64_t N = *in0_shape.rbegin();
    int64_t K = *++in0_shape.rbegin();
    int64_t n_blk_size = N > 64 ? 64 : N;
    config.update(N, K, n_blk_size);

    std::cout << "biasMem.resize to N:" << N << std::endl;
    biasMem.resize(N * sizeof(float), 0);
}

// for K*N(32*512) and nb(n_block-64), repack each nb block(32*64) to nbK8nb.
void GemmCopyBKaiKernelExecutor::execute(const GemmCopyBKaiKernelExecutor* executor, void* in0, void* out0) {
    // return;
    std::cout << "copyb in:" << reinterpret_cast<size_t>(in0) << std::endl;
    std::cout << "copyb out:" << reinterpret_cast<size_t>(out0) << std::endl;
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    std::cout << "GemmCopyBKaiKernelExecutor excute" << std::endl;
    // rhs-in, rhs_packed-out
    // params: N, K, n_blk_size. 8 is fixed.
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto K = config.get_K();            // K
    const auto N = config.get_N();            // N-rhs_stride
    const auto& n_blk_size = config.get_n();  // n_blk
    std::cout << "K:" << K << std::endl;
    std::cout << "N:" << N << std::endl;
    std::cout << "n_blk_size:" << n_blk_size << std::endl;
    // float* src_ptr_test = static_cast<float*>(in0);
    // for (size_t i = 0; i < K * N; i++) {
    //     std::cout << "inp_i:" << i << " value:" << src_ptr_test[i] << std::endl;
    // }
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    size_t n_blocks = (N + n_blk_size - 1) / n_blk_size;
    size_t dst_offset = 0;
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * n_blk_size;
        size_t n_end = std::min(n_start + n_blk_size, static_cast<size_t>(N));
        size_t n_block_size = n_end - n_start;
        // size_t in_oc = kai_get_rhs_offset_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_start);
        // size_t pa_oc = kai_get_rhs_packed_offset_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_block_size, 0);
        // int8_t* src_ptr = static_cast<int8_t*>(in0) + in_oc;
        // int8_t* dst_ptr = static_cast<int8_t*>(out0) + pa_oc;
        std::cout << "n_block:" << n_block << std::endl;
        std::cout << "n_start:" << n_start << std::endl;
        std::cout << "n_end:" << n_end << std::endl;
        std::cout << "n_block_size:" << n_block_size << std::endl;
        // std::cout << "in_oc:" << in_oc << std::endl;
        // std::cout << "pa_oc:" << pa_oc << std::endl;
        int8_t* src_ptr = static_cast<int8_t*>(in0) + n_start * sizeof(int32_t);
        // int8_t* dst_ptr = static_cast<int8_t*>(out0) + (n_start * K * sizeof(int32_t));
        int8_t* dst_ptr = static_cast<int8_t*>(out0) + dst_offset;
        const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_block_size, K);
        dst_offset += rhsPackedSize;
        std::cout << "copyb n_block:" << n_block << std::endl;
        std::cout << "copyb out block address:" << reinterpret_cast<size_t>(dst_ptr) << std::endl;
        std::cout << "N" << n_block_size << std::endl;
        std::cout << "K" << K << std::endl;
        std::cout << "nr" << nr << std::endl;
        std::cout << "kr" << kr << std::endl;
        std::cout << "sr" << sr << std::endl;
        std::cout << "rhs_stride" << N * sizeof(float) << std::endl;
        std::cout << "rhsPackedSize" << rhsPackedSize << std::endl;
        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                         n_block_size,
                                                         K,
                                                         nr,
                                                         kr,
                                                         sr,                        // Packing arguments
                                                         N * sizeof(float),         // RHS stride
                                                         src_ptr,                   // RHS
                                                         executor->get_bias_mem(),  // Bias
                                                         nullptr,                   // Scale
                                                         dst_ptr,                   // RHS packed
                                                         0,
                                                         nullptr);
    }
    // float* dst_ptr_test = static_cast<float*>(out0);
    // for (size_t i = 0; i < K * N; i++) {
    //     std::cout << "out_i:" << i << " value:" << dst_ptr_test[i] << std::endl;
    // }
}

}  // namespace ov::intel_cpu::aarch64
