// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "mvn_config.hpp"

namespace ov {
namespace intel_cpu {

class MVNExecutor {
public:
    MVNExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const MVNAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) = 0;
    virtual ~MVNExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    MVNAttrs mvnAttrs;
    const ExecutorContext::CPtr context;
};

using MVNExecutorPtr = std::shared_ptr<MVNExecutor>;
using MVNExecutorCPtr = std::shared_ptr<const MVNExecutor>;

class MVNExecutorBuilder {
public:
    ~MVNExecutorBuilder() = default;
    virtual bool isSupported(const MVNAttrs& mvnAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual MVNExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using MVNExecutorBuilderPtr = std::shared_ptr<MVNExecutorBuilder>;
using MVNExecutorBuilderCPtr = std::shared_ptr<const MVNExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov
