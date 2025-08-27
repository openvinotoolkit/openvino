// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>
#include <vector>

#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/x64/jit_matmul_small.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct MatMulSmallAttrs {
    size_t M = 0UL;
    size_t N = 0UL;
    size_t K = 0UL;
    size_t WA = 0UL;
};

class MatMulSmallExecutor : public Executor {
public:
    MatMulSmallExecutor(const MatMulAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        // @todo distinguish cpu plugin jit from dnnl jit
        return impl_desc_type::unknown;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const MatMulConfig& config);

private:
    // set post_ops_args based on primArgs and post_ops
    void prepare_binary_args(const DnnlPrimitiveAttrs& primAttrs);

    DnnlShapeAgnosticDataPtr shapeAgnosticData;
    MatMulSmallAttrs m_matmul_attrs;
    std::shared_ptr<jit_uni_matmul_small_kernel> m_matmul_kernel;
    std::vector<const void*> m_post_ops_args;
};

}  // namespace ov::intel_cpu
