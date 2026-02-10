// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"

namespace ov::intel_cpu {

struct ConcatAttrs {
    size_t axis = 0;
};

class ConcatExecutor {
public:
    explicit ConcatExecutor(ExecutorContext::CPtr context);
    virtual ~ConcatExecutor() = default;

    virtual bool init(const ConcatAttrs& concatAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) = 0;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    ConcatAttrs concatAttrs;
    ExecutorContext::CPtr context;
};

using ConcatExecutorPtr = std::shared_ptr<ConcatExecutor>;
using ConcatExecutorCPtr = std::shared_ptr<const ConcatExecutor>;

class ConcatExecutorBuilder {
public:
    virtual ~ConcatExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const ConcatAttrs& concatAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual ConcatExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using ConcatExecutorBuilderPtr = std::shared_ptr<ConcatExecutorBuilder>;
using ConcatExecutorBuilderCPtr = std::shared_ptr<const ConcatExecutorBuilder>;

struct ConcatExecutorDesc {
    ExecutorType executorType;
    ConcatExecutorBuilderCPtr builder;
};

}  // namespace ov::intel_cpu
