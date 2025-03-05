// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Tensor.h"
#include "nodes/executors/interpolate.hpp"

namespace ov::intel_cpu {

class ACLInterpolateExecutor : public InterpolateExecutor {
public:
    ACLInterpolateExecutor(const ExecutorContext::CPtr context) : InterpolateExecutor(context) {}

    bool init(const InterpolateAttrs& interpolateAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;
    InterpolateAttrs aclInterpolateAttrs;
    arm_compute::SamplingPolicy acl_coord;
    arm_compute::InterpolationPolicy acl_policy;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEScale> acl_scale;
};

class ACLInterpolateExecutorBuilder : public InterpolateExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const InterpolateAttrs& interpolateAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] InterpolateExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLInterpolateExecutor>(context);
    }

private:
    static bool isSupportedConfiguration(const InterpolateAttrs& interpolateAttrs,
                                         const std::vector<MemoryDescPtr>& srcDescs,
                                         const std::vector<MemoryDescPtr>& dstDescs);
};
}  // namespace ov::intel_cpu
