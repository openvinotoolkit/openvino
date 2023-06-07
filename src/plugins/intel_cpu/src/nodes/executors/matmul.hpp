// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
    bool withBias;
};

class MatMulExecutor {
public:
    MatMulExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const MatMulAttrs& mvnAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~MatMulExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    MatMulAttrs mvnAttrs;
    const ExecutorContext::CPtr context;
};

using MatMulExecutorPtr = std::shared_ptr<MatMulExecutor>;
using MatMulExecutorCPtr = std::shared_ptr<const MatMulExecutor>;

class MatMulExecutorBuilder {
public:
    virtual ~MatMulExecutorBuilder() = default;
    virtual bool isSupported(const MatMulAttrs& MatMulAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) const = 0;
    virtual MatMulExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using MatMulExecutorBuilderPtr = std::shared_ptr<MatMulExecutorBuilder>;
using MatMulExecutorBuilderCPtr = std::shared_ptr<const MatMulExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov
