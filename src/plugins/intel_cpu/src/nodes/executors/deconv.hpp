// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

struct DeconvKey {
    DnnlMemoryDescCPtr inp0;
    DnnlMemoryDescCPtr inp1;
    DnnlMemoryDescCPtr bias;
    DnnlMemoryDescCPtr out;

    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    ov::CoordinateDiff paddingL;
    ov::CoordinateDiff paddingR;

    bool isInt8;

    dnnl::primitive_attr attr;
    impl_desc_type implType;

    size_t hash() const;
    bool operator==(const DeconvKey& rhs) const;
};

// Defines way to add epsilon: inside sqrt or outside.
struct DeconvAttrs {
    bool withBiases = false;
    std::vector<ptrdiff_t> kernel;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;
    ov::CoordinateDiff outputPadding;
    std::vector<int32_t> lastOutputSpatialDims;
    VectorDims int8WeightDims;
    VectorDims expectedBiasDims;
    bool withGroups = false;
    bool isDW = false;
    bool isInt8 = false;
    bool autoPad = false;
    bool externOutShape = false;
    size_t groupNum = 1;
    size_t IC;
    size_t OC;
    dnnl::engine engine;
    DeconvKey key;
    std::unordered_map<int, dnnl::memory> primArgs;
    MultiCachePtr cache;
    std::function<void()> deconvAppendPostOpArgs;
    std::function<MemoryPtr(DnnlMemoryDescPtr)> deconvGetScratchPadMem;
    MemoryPtr biasMemPtr = nullptr;
    std::vector<MemoryPtr> *deconvInternalBlobMemory;
    std::string layerName;
};

class DeconvExecutor {
public:
    DeconvExecutor();
    virtual bool init(const DeconvAttrs& deconvAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void *post_ops_data_,
                      dnnl::stream strm,
                      std::vector<MemoryPtr> internalBlobMemory = std::vector<MemoryPtr>()) = 0;
    virtual ~DeconvExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    DeconvAttrs deconvAttrs;
};

using DeconvExecutorPtr = std::shared_ptr<DeconvExecutor>;
using DeconvExecutorCPtr = std::shared_ptr<const DeconvExecutor>;

class DeconvExecutorBuilder {
public:
    ~DeconvExecutorBuilder() = default;
    virtual bool isSupported(const DeconvAttrs& convAttrs, const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual DeconvExecutorPtr makeExecutor() const = 0;
};

using DeconvExecutorBuilderPtr = std::shared_ptr<DeconvExecutorBuilder>;
using DeconvExecutorBuilderCPtr = std::shared_ptr<const DeconvExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov
