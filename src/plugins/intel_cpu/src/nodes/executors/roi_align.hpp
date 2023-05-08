// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "dnnl_scratch_pad.h"
#include "executor.hpp"
#include <ngraph/opsets/opset9.hpp>

namespace ov {
namespace intel_cpu {

using ngPoolingMode = ngraph::opset9::ROIAlign::PoolingMode;

enum ROIAlignedMode {
    ra_asymmetric,
    ra_half_pixel_for_nn,
    ra_half_pixel
};

struct ROIAlignAttrs {
    int pooledH = 7;
    int pooledW = 7;
    int samplingRatio = 2;
    float spatialScale = 1.0f;
    ROIAlignedMode alignedMode;
    ngPoolingMode m;
};

class ROIAlignExecutor {
public:
    ROIAlignExecutor(const ExecutorContext::CPtr context);
    virtual bool init(const ROIAlignAttrs& roialignAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;

    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) = 0;
    virtual ~ROIAlignExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    ROIAlignAttrs roialignAttrs;
    const ExecutorContext::CPtr context;
};

using ROIAlignExecutorPtr = std::shared_ptr<ROIAlignExecutor>;
using ROIAlignExecutorCPtr = std::shared_ptr<const ROIAlignExecutor>;

class ROIAlignExecutorBuilder {
public:
    ~ROIAlignExecutorBuilder() = default;
    virtual bool isSupported(const ROIAlignAttrs& roialignAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual ROIAlignExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using ROIAlignExecutorBuilderPtr = std::shared_ptr<ROIAlignExecutorBuilder>;
using ROIAlignExecutorBuilderCPtr = std::shared_ptr<const ROIAlignExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov