// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/permute_kernel.h"

namespace ov::intel_cpu {

struct TransposeParams {
    PermuteParams permuteParams;
};

class TransposeExecutor : public Executor {
public:
    static jit_permute_config_params prepareParams(const PermuteParams& params);
    explicit TransposeExecutor(ExecutorContext::CPtr context);
    virtual bool init(const TransposeParams& transposeParams,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;
    ~TransposeExecutor() override = default;

protected:
    PermuteParams permuteParams;
    const ExecutorContext::CPtr context;
};
using TransposeExecutorPtr = std::shared_ptr<TransposeExecutor>;
using TransposeExecutorCPtr = std::shared_ptr<const TransposeExecutor>;

class TransposeExecutorBuilder {
public:
    virtual ~TransposeExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const TransposeParams& transposeParams,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual TransposeExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using TransposeExecutorBuilderPtr = std::shared_ptr<TransposeExecutorBuilder>;
using TransposeExecutorBuilderCPtr = std::shared_ptr<const TransposeExecutorBuilder>;

}  // namespace ov::intel_cpu
