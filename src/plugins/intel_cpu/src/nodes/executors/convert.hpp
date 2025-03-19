// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "executor.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct ConvertParams {
    ov::element::Type srcPrc;
    ov::element::Type origPrc;
    ov::element::Type dstPrc;
    size_t size;
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
    [[nodiscard]] virtual ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ConvertExecutorBuilderPtr = std::shared_ptr<ConvertExecutorBuilder>;
using ConvertExecutorBuilderCPtr = std::shared_ptr<const ConvertExecutorBuilder>;

}  // namespace ov::intel_cpu
