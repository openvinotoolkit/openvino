// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_types.h"
#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"

namespace ov::intel_cpu {

struct SplitAttrs {
    size_t axis = 0;
};

class SplitExecutor {
public:
    explicit SplitExecutor(ExecutorContext::CPtr context);
    virtual ~SplitExecutor() = default;

    virtual bool init(const SplitAttrs& splitAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) = 0;

    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;

protected:
    SplitAttrs splitAttrs;
    const ExecutorContext::CPtr context;
};

using SplitExecutorPtr = std::shared_ptr<SplitExecutor>;
using SplitExecutorCPtr = std::shared_ptr<const SplitExecutor>;

class SplitExecutorBuilder {
public:
    virtual ~SplitExecutorBuilder() = default;

    [[nodiscard]] virtual bool isSupported(const SplitAttrs& splitAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;

    [[nodiscard]] virtual SplitExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using SplitExecutorBuilderPtr = std::shared_ptr<SplitExecutorBuilder>;
using SplitExecutorBuilderCPtr = std::shared_ptr<const SplitExecutorBuilder>;

}  // namespace ov::intel_cpu
