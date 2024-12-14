// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "executor.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

struct ConvertParams {
    ov::element::Type srcPrc;
    ov::element::Type origPrc;
    ov::element::Type dstPrc;
    size_t size;
};

class ConvertExecutor : public Executor {
public:
    explicit ConvertExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ConvertParams& convertParams,
                      const MemoryDescPtr& srcDesc,
                      const MemoryDescPtr& dstDesc,
                      const dnnl::primitive_attr& attr) = 0;
    virtual ~ConvertExecutor() = default;

protected:
    ConvertParams convertParams;
    const ExecutorContext::CPtr convertContext;
};
using ConvertExecutorPtr = std::shared_ptr<ConvertExecutor>;
using ConvertExecutorCPtr = std::shared_ptr<const ConvertExecutor>;

class ConvertExecutorBuilder {
public:
    virtual ~ConvertExecutorBuilder() = default;
    virtual bool isSupported(const ConvertParams& convertParams,
                             const MemoryDescPtr& srcDesc,
                             const MemoryDescPtr& dstDesc) const = 0;
    virtual ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ConvertExecutorBuilderPtr = std::shared_ptr<ConvertExecutorBuilder>;
using ConvertExecutorBuilderCPtr = std::shared_ptr<const ConvertExecutorBuilder>;

}  // namespace intel_cpu
}  // namespace ov
