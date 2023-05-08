// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "dnnl_scratch_pad.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct ShuffleChannelsAttributes {
    LayoutType layoutType;
    int dataRank = 0;
    int axis = 0;
    int spatialRank = 0;
    size_t group = 0lu;
    size_t dataSize = 1lu;
    VectorDims srcDims;
    VectorDims srcBlockedDims;
    impl_desc_type implDescType;
    size_t hash() const;
    bool operator==(const ShuffleChannelsAttributes& rhs) const;
};

class ShuffleChannelsExecutor {
public:
    explicit ShuffleChannelsExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) = 0;
    virtual impl_desc_type getImplType() const = 0;
    virtual ~ShuffleChannelsExecutor() = default;
protected:
    const ExecutorContext::CPtr shuffleChannelsContext;
};

using ShuffleChannelsExecutorPtr = std::shared_ptr<ShuffleChannelsExecutor>;
using ShuffleChannelsExecutorCPtr = std::shared_ptr<const ShuffleChannelsExecutor>;

class ShuffleChannelsExecutorBuilder {
public:
    ~ShuffleChannelsExecutorBuilder() = default;
    virtual bool isSupported(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual ShuffleChannelsExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ShuffleChannelsExecutorBuilderPtr = std::shared_ptr<ShuffleChannelsExecutorBuilder>;
using ShuffleChannelsExecutorBuilderCPtr = std::shared_ptr<const ShuffleChannelsExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov