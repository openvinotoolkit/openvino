// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/roi_align.hpp"
#include "utils/debug_capabilities.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class AclROIAlignExecutor : public ROIAlignExecutor {
public:
    AclROIAlignExecutor(const ExecutorContext::CPtr context);

    bool init(const ROIAlignAttrs& roialignAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    ROIAlignAttrs roialignAttrs;
    std::vector<float> roiBuffer;
    int numRois;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor roiTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEROIAlignLayer> roialign = nullptr;
};

class AclROIAlignExecutorBuilder : public ROIAlignExecutorBuilder {
public:
    bool isSupported(const ROIAlignAttrs& roialignAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
            srcDescs[0]->getPrecision() != srcDescs[1]->getPrecision() ||
           (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
            srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16))
            return false;
        //ACL supports only AVG and asymmetric/half pixel aligned mode
        if (roialignAttrs.m != ngPoolingMode::AVG ||
            (roialignAttrs.alignedMode != ROIAlignedMode::ra_asymmetric &&
            roialignAttrs.alignedMode != ROIAlignedMode::ra_half_pixel))
            return false;
        return true;
    }

    ROIAlignExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclROIAlignExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov