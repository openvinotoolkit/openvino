// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_gemm.hpp"

#include "emitters/tpp/common/utils.hpp"
#include "openvino/core/parallel.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu::tpp {

BrgemmKaiKernelExecutor::BrgemmKaiKernelExecutor(BrgemmKernelKaiConfig config)
    : snippets::KernelExecutor<BrgemmKernelKaiConfig, BrgemmTppKaiCompiledKernel>(std::move(config)) {}

void BrgemmKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                            const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                            BrgemmKernelKaiConfig& config) const {
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

    // allocate
    rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    auto threads = parallel_get_max_threads();
    rhsPackedMem.resize(rhsPackedSize * sizeof(float) * threads, 0);
    biasMem.resize(N * sizeof(float), 0);
}

void BrgemmKaiKernelExecutor::execute(const BrgemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // process subtensor. k should not be blocked as kai doesn't support beta == 1.
    // repack input2, directly done here and no copyB needed. ok? where to allocate memory for repacked input?
    // matmul for input1 and slices of repacked input2
    const auto& config = static_cast<const BrgemmKernelKaiConfig&>(executor->get_config());
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
    kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                     N,
                                                     K,
                                                     nr,
                                                     kr,
                                                     sr,                        // Packing arguments
                                                     rhs_stride,                // RHS stride
                                                     in1,                       // RHS
                                                     executor->get_bias_mem(),  // Bias
                                                     nullptr,                   // Scale
                                                     packed,                    // RHS packed
                                                     0,
                                                     nullptr);

    const size_t BLOCK_SIZE = 8;
    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t lhs_stride = lda * sizeof(float);  // if K not split, it's K * sizeof(float)
    const size_t dst_stride_row = ldc * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * BLOCK_SIZE;
        size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
        size_t n_block_size = n_end - n_start;
        const size_t rhs_packed_offset =
            ukernel.get_rhs_packed_offset(n_start, K);  // should be K, as packed mem as 8*K blocks
        const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
        const float* rhs_ptr = packed + rhs_packed_offset / sizeof(float);
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

}  // namespace ov::intel_cpu::tpp
