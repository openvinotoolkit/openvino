// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct ConvertParams {
    InferenceEngine::Precision srcPrc;
    InferenceEngine::Precision origPrc;
    InferenceEngine::Precision dstPrc;
    size_t size;
};

class ConvertExecutor {
public:
    explicit ConvertExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ConvertParams& convertParams,
                      const MemoryDescPtr& srcDesc,
                      const MemoryDescPtr& dstDesc,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const MemoryCPtr& src, const MemoryPtr& dst) = 0;
    virtual impl_desc_type getImplType() const = 0;
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

} // namespace intel_cpu
} // namespace ov