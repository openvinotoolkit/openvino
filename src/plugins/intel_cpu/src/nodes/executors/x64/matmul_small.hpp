// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "jit_matmul_small.hpp"
#include "nodes/common/dnnl_executor.h"

namespace ov::intel_cpu {

struct MatMulSmallAttrs {
    size_t M;
    size_t N;
    size_t K;
};

class MatMulSmallExecutor : public DnnlExecutorLegacy {
public:
    MatMulSmallExecutor(const MatMulSmallAttrs& matmulAttrs, const dnnl::primitive_desc& pd);
    void exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm) override;
    virtual ~MatMulSmallExecutor() = default;

protected:
    MatMulSmallAttrs matmulAttrs;
    std::shared_ptr<jit_uni_matmul_small_kernel> matmul_kernel;
};

}  // namespace ov::intel_cpu
