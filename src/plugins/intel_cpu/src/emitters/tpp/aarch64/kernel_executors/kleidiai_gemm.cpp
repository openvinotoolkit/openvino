// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kleidiai_gemm.hpp"

#include "emitters/tpp/common/utils.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

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
    const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(N, K);
    // auto rhsPackedDesc = std::make_shared<CpuBlockedMemoryDesc>(f32, Shape({rhsPackedSize}));
    // rhsPackedMem = std::make_shared<Memory>(context->getEngine(), rhsPackedDesc);
    rhsPackedMem.resize(rhsPackedSize * sizeof(float));
}

void BrgemmKaiKernelExecutor::execute(const BrgemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // libxsmm_gemm_param gemm_p;
    // // data is row major, but libxsmm gemm suppose column major. in0 and in1 are exchanged to avoid data repack(kernel
    // // creation params aligned).
    // gemm_p.a.primary = in1;
    // gemm_p.b.primary = in0;
    // gemm_p.c.primary = out0;
    // auto brg_kernel = executor->get_kernel();
    // OV_CPU_JIT_EMITTER_ASSERT(brg_kernel, "has nullptr compiler kernel");
    // OV_CPU_JIT_EMITTER_ASSERT(brg_kernel->brgemm_kernel, "has nullptr compiler brgemm_kernel");
    // (*(brg_kernel->brgemm_kernel))(&gemm_p);

    // process subtensor. k should not be blocked as kai doesn't support beta == 1.
    // repack input2 once, directly done here and no copyB needed. ok? where to allocate memory for repacked input?
    // matmul for input1 and slices of input2
    // auto config = get_config();
    size_t BLOCK_SIZE = 8;
    // size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // for (size_t block_num = 0; block_num < n_blocks; block_num++) {

    // }
}

}  // namespace ov::intel_cpu::tpp
