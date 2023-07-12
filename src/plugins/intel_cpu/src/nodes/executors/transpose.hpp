// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov {
namespace intel_cpu {

struct TransposeParams {
    PermuteParams permuteParams;
};

class TransposeExecutor {
public:
    static jit_permute_config_params prepareParams(const PermuteParams& params);
    explicit TransposeExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const TransposeParams& transposeParams,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) = 0;
    virtual impl_desc_type getImplType() const = 0;
    virtual ~TransposeExecutor() = default;
protected:
    PermuteParams permuteParams;
    const ExecutorContext::CPtr context;
};
using TransposeExecutorPtr = std::shared_ptr<TransposeExecutor>;
using TransposeExecutorCPtr = std::shared_ptr<const TransposeExecutor>;

class TransposeExecutorBuilder {
public:
    virtual ~TransposeExecutorBuilder() = default;
    virtual bool isSupported(const TransposeParams& transposeParams,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using TransposeExecutorBuilderPtr = std::shared_ptr<TransposeExecutorBuilder>;
using TransposeExecutorBuilderCPtr = std::shared_ptr<const TransposeExecutorBuilder>;

} // namespace intel_cpu
} // namespace ov