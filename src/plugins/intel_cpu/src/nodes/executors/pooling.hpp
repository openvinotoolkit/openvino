// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct PoolingAttrs {
    bool exclude_pad = false;
    bool auto_pad = false;

    op::PadType pad_type;
    Algorithm algorithm;

    op::RoundingType rounding;

    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> dilation;

    std::vector<ptrdiff_t> data_pad_begin;
    std::vector<ptrdiff_t> data_pad_end;

    /// Effective padding. Used to define correct output shape by oneDNN
    /// reshape formula: (iw - kernel + pad_l + pad_r) / strides[i - 2] + 1
    /// should be passed into pooling desc constructor.
    std::vector<ptrdiff_t> effective_pad_begin;
    std::vector<ptrdiff_t> effective_pad_end;

    /// Effective dilation. Used to define correct dilation for OneDNN.
    /// For OneDNN default dilation is vector of zero
    std::vector<ptrdiff_t> effective_dilation;
};

class PoolingExecutor {
public:
    PoolingExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const PoolingAttrs& poolingAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~PoolingExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    PoolingAttrs poolingAttrs;
    const ExecutorContext::CPtr context;
};

using PoolingExecutorPtr = std::shared_ptr<PoolingExecutor>;
using PoolingExecutorCPtr = std::shared_ptr<const PoolingExecutor>;

class PoolingExecutorBuilder {
public:
    ~PoolingExecutorBuilder() = default;
    virtual bool isSupported(const PoolingAttrs& poolingAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual PoolingExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using PoolingExecutorBuilderPtr = std::shared_ptr<PoolingExecutorBuilder>;
using PoolingExecutorBuilderCPtr = std::shared_ptr<const PoolingExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov