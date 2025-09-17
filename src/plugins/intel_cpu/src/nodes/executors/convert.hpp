// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

struct ConvertParams {
    ov::element::Type srcPrc;
    ov::element::Type origPrc;
    ov::element::Type dstPrc;
    bool no_clamp = false;
    bool use_rounding = false;
    size_t size = 0UL;
};

class ConvertExecutor : public Executor {
public:
    explicit ConvertExecutor(ExecutorContext::CPtr context);
    virtual bool init(const ConvertParams& convertParams,
                      const MemoryDescPtr& srcDesc,
                      const MemoryDescPtr& dstDesc,
                      const dnnl::primitive_attr& attr) = 0;
    ~ConvertExecutor() override = default;

protected:
    ConvertParams convertParams;
    const ExecutorContext::CPtr convertContext;
};
using ConvertExecutorPtr = std::shared_ptr<ConvertExecutor>;
using ConvertExecutorCPtr = std::shared_ptr<const ConvertExecutor>;

class ConvertExecutorBuilder {
public:
    virtual ~ConvertExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const ConvertParams& convertParams,
                                           const MemoryDescPtr& srcDesc,
                                           const MemoryDescPtr& dstDesc) const = 0;
    [[nodiscard]] virtual ConvertExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using ConvertExecutorBuilderPtr = std::shared_ptr<ConvertExecutorBuilder>;
using ConvertExecutorBuilderCPtr = std::shared_ptr<const ConvertExecutorBuilder>;

}  // namespace ov::intel_cpu
