// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "nodes/executors/matmul.hpp"
#include "nodes/kernels/x64/jit_matmul_small.hpp"

namespace ov::intel_cpu {

struct MatMulSmallAttrs {
    size_t M = 0UL;
    size_t N = 0UL;
    size_t K = 0UL;
    dnnl::primitive_attr attr;
};

class MatMulSmallExecutor : public IMatmulExecutor {
public:
    MatMulSmallExecutor(MatMulSmallAttrs attrs);
    void exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm) override;
    [[nodiscard]] DnnlMemoryDescPtr getScratchPadDesc() const override;

private:
    // set post_ops_args based on primArgs and post_ops
    void prepare_binary_args(const std::unordered_map<int, dnnl::memory>& primArgs);

    MatMulSmallAttrs m_matmul_attrs;
    std::shared_ptr<jit_uni_matmul_small_kernel> m_matmul_kernel;
    std::vector<const void*> m_post_ops_args;
};

}  // namespace ov::intel_cpu
