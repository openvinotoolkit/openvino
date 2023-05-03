// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"

namespace ov {
namespace intel_cpu {

struct SpaceToDepthAttrs {
    LayoutType layoutType;
    enum Mode { BLOCKS_FIRST = 0, DEPTH_FIRST = 1 } mode;
    size_t blockSize = 0lu;
    size_t blockStep = 1lu;
    size_t dataSize = 1lu;
    size_t nSpatialDims = 0lu;
    VectorDims srcBlockedDims;
    VectorDims destBlockedDims;
    size_t hash() const;
    bool operator==(const SpaceToDepthAttrs& rhs) const;
    impl_desc_type implDescType;
};

class SpaceToDepthExecutor {
public:
    explicit SpaceToDepthExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const SpaceToDepthAttrs& spaceToDepthAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) = 0;
    virtual impl_desc_type getImplType() const = 0;
    virtual ~SpaceToDepthExecutor() = default;
protected:
    const ExecutorContext::CPtr spaceToDepthContext;
};

using SpaceToDepthExecutorPtr = std::shared_ptr<SpaceToDepthExecutor>;
using SpaceToDepthExecutorCPtr = std::shared_ptr<const SpaceToDepthExecutor>;

class SpaceToDepthExecutorBuilder {
public:
    ~SpaceToDepthExecutorBuilder() = default;
    virtual bool isSupported(const SpaceToDepthAttrs& spaceToDepthAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual SpaceToDepthExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using SpaceToDepthExecutorBuilderPtr = std::shared_ptr<SpaceToDepthExecutorBuilder>;
using SpaceToDepthExecutorBuilderCPtr = std::shared_ptr<const SpaceToDepthExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov