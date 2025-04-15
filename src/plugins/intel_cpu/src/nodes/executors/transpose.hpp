// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "executor.hpp"
#include "nodes/common/permute_kernel.h"
#include "onednn/iml_type_mapper.h"

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
    [[nodiscard]] virtual TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using TransposeExecutorBuilderPtr = std::shared_ptr<TransposeExecutorBuilder>;
using TransposeExecutorBuilderCPtr = std::shared_ptr<const TransposeExecutorBuilder>;

}  // namespace ov::intel_cpu
