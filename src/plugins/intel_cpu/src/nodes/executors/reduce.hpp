// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "dnnl_scratch_pad.h"
#include "executor.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct ReduceAttrs {
    std::vector<int> axes;
    Algorithm operation;
    bool keepDims;
};

class ReduceExecutor {
public:
    ReduceExecutor(ExecutorContext::CPtr context);
    virtual bool init(const ReduceAttrs& reduceAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    virtual ~ReduceExecutor() = default;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    ReduceAttrs reduceAttrs;
    const ExecutorContext::CPtr context;
};

using ReduceExecutorPtr = std::shared_ptr<ReduceExecutor>;
using ReduceExecutorCPtr = std::shared_ptr<const ReduceExecutor>;

class ReduceExecutorBuilder {
public:
    ~ReduceExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const ReduceAttrs& reduceAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ReduceExecutorBuilderPtr = std::shared_ptr<ReduceExecutorBuilder>;
using ReduceExecutorBuilderCPtr = std::shared_ptr<const ReduceExecutorBuilder>;

}  // namespace ov::intel_cpu
