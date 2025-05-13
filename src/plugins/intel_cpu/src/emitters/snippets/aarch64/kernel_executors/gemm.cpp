// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm.hpp"

// #include "emitters/tpp/common/utils.hpp"
#include "openvino/core/parallel.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu::aarch64 {

GemmKaiKernelExecutor::GemmKaiKernelExecutor(GemmKernelKaiConfig config)
    : snippets::KernelExecutor<GemmKernelKaiConfig, GemmKaiCompiledKernel>(std::move(config)) {}

void GemmKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                          const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                          GemmKernelKaiConfig& config) const {
    // update M/N/K/beta
    int64_t M, N, K, beta;
    std::tie(M, N, K, beta) = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);

    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDC = snippets::utils::get_dim_stride(expr->get_output_port(0));
    const auto LDB =
        snippets::utils::get_dim_stride(expr->get_input_port(1));  // no (external) repack for second input for now

    config.update(M, N, K, LDA, LDB, LDC, beta);

    // allocate
    rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    auto threads = parallel_get_max_threads();
    rhsPackedMem.resize(rhsPackedSize * sizeof(float) * threads, 0);
    biasMem.resize(N * sizeof(float), 0);
}

// void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
//     std::cout << "GemmKaiKernelExecutor::execute" << std::endl;
//     OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
//     // process subtensor. k should not be blocked as kai doesn't support beta == 1.
//     // repack input2, directly done here and no copyB needed. ok? where to allocate memory for repacked input?
//     // matmul for input1 and slices of repacked input2
//     const auto& config = static_cast<const GemmKernelKaiConfig&>(executor->get_config());
//     const auto& M = config.get_M();
//     const auto& N = config.get_N();
//     const auto& K = config.get_K();
//     const auto& lda = config.get_LDA();
//     const auto& ldb = config.get_LDB();
//     const auto& ldc = config.get_LDC();
//     const size_t nr = ukernel.get_nr();
//     const size_t kr = ukernel.get_kr();
//     const size_t sr = ukernel.get_sr();
//     const size_t rhs_stride = ldb * sizeof(float);
//     auto thread_idx = parallel_get_thread_num();
//     auto packed = static_cast<float*>(executor->get_packed_mem());
//     packed += thread_idx * executor->rhsPackedSize;
//     float* src_ptr_test = static_cast<float*>(in1);
//     // for (size_t i = 0; i < K * N; i++) {
//     //     std::cout << "inp_i:" << i << " value:" << src_ptr_test[i] << std::endl;
//     // }
//     std::cout << "N" << N << std::endl;
//     std::cout << "K" << K << std::endl;
//     std::cout << "nr" << nr << std::endl;
//     std::cout << "kr" << kr << std::endl;
//     std::cout << "sr" << sr << std::endl;
//     std::cout << "rhs_stride" << rhs_stride << std::endl;
//     kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
//                                                      N,
//                                                      K,
//                                                      nr,
//                                                      kr,
//                                                      sr,                        // Packing arguments
//                                                      rhs_stride,                // RHS stride
//                                                      in1,                       // RHS
//                                                      executor->get_bias_mem(),  // Bias
//                                                      nullptr,                   // Scale
//                                                      packed,                    // RHS packed
//                                                      0,
//                                                      nullptr);
//     float* dst_ptr_test = static_cast<float*>(packed);
//     // for (size_t i = 0; i < K * N; i++) {
//     //     std::cout << "out_i:" << i << " value:" << dst_ptr_test[i] << std::endl;
//     // }

//     const size_t BLOCK_SIZE = 8;
//     size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     const size_t lhs_stride = lda * sizeof(float);  // if K not split, it's K * sizeof(float)
//     const size_t dst_stride_row = ldc * sizeof(float);
//     const size_t dst_stride_col = sizeof(float);
//     for (size_t n_block = 0; n_block < n_blocks; n_block++) {
//         size_t n_start = n_block * BLOCK_SIZE;
//         size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
//         size_t n_block_size = n_end - n_start;
//         const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(
//             n_start,
//             K);  // result is n_start*k. should be K, as packed mem as 8*K blocks. If k blocked, then lda.
//         const size_t dst_offset =
//             ukernel.get_dst_offset(0, n_start, dst_stride_row);  // m_idx is 0 as dst already point current block
//         const float* rhs_ptr = packed + rhs_packed_offset / sizeof(float);
//         float* dst_ptr = (static_cast<float*>(out0) + dst_offset / (sizeof(float)));
//         ukernel.run_matmul(M,
//                            n_block_size,
//                            K,
//                            in0,
//                            lhs_stride,
//                            rhs_ptr,
//                            dst_ptr,
//                            dst_stride_row,
//                            dst_stride_col,
//                            FLOAT_MIN,
//                            FLOAT_MAX);
//     }
// }

void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
    // return;
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    std::cout << "GemmKaiKernelExecutor in1:" << reinterpret_cast<size_t>(in1) << std::endl;
    // process subtensor. k should not be blocked as kai doesn't support beta == 1.
    // repack input2, directly done here and no copyB needed. ok? where to allocate memory for repacked input?
    // matmul for input1 and slices of repacked input2
    const auto& config = static_cast<const GemmKernelKaiConfig&>(executor->get_config());
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldb = config.get_LDB();
    const auto& ldc = config.get_LDC();
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    const size_t rhs_stride = ldb * sizeof(float);
    auto thread_idx = parallel_get_thread_num();
    auto packed = static_cast<float*>(executor->get_packed_mem());
    packed += thread_idx * executor->rhsPackedSize;
    // kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
    //                                                  N,
    //                                                  K,
    //                                                  nr,
    //                                                  kr,
    //                                                  sr,                        // Packing arguments
    //                                                  rhs_stride,                // RHS stride
    //                                                  in1,                       // RHS
    //                                                  executor->get_bias_mem(),  // Bias
    //                                                  nullptr,                   // Scale
    //                                                  packed,                    // RHS packed
    //                                                  0,
    //                                                  nullptr);s

    const size_t BLOCK_SIZE = 8;
    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t lhs_stride = lda * sizeof(float);  // if K not split, it's K * sizeof(float)
    const size_t dst_stride_row = ldc * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * BLOCK_SIZE;
        size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
        size_t n_block_size = n_end - n_start;
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(
            n_start,
            K);  // result is n_start*k. should be K, as packed mem as 8*K blocks. If k blocked, then lda.
        const size_t dst_offset =
            ukernel.get_dst_offset(0, n_start, dst_stride_row);  // m_idx is 0 as dst already point current block
        // const float* rhs_ptr = packed + rhs_packed_offset / sizeof(float);
        // in0, in1, out is point to current block memory, based on block loop info, and done in loop begin and end
        // emitters(adjust copyb loop info after repack outside block loops).
        float* rhs_ptr = static_cast<float*>(in1) + rhs_packed_offset / sizeof(float);
        float* dst_ptr = (static_cast<float*>(out0) + dst_offset / (sizeof(float)));
        ukernel.run_matmul(M,
                           n_block_size,
                           K,
                           in0,
                           lhs_stride,
                           rhs_ptr,
                           dst_ptr,
                           dst_stride_row,
                           dst_stride_col,
                           FLOAT_MIN,
                           FLOAT_MAX);
    }
}

}  // namespace ov::intel_cpu::aarch64
