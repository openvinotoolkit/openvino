// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/Tensor.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"

namespace ov::intel_cpu {

class AclInterpolateExecutor : public Executor {
public:
    AclInterpolateExecutor(InterpolateAttrs attrs, const MemoryArgs& /*memory*/, ExecutorContext::CPtr context)
        : aclInterpolateAttrs(std::move(attrs)), context(std::move(context)) {}

    static bool supports(const InterpolateConfig& config);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override { return impl_desc_type::acl; }

private:
    InterpolateAttrs aclInterpolateAttrs;
    const ExecutorContext::CPtr context;
    arm_compute::SamplingPolicy acl_coord = arm_compute::SamplingPolicy::CENTER;
    arm_compute::InterpolationPolicy acl_policy = arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEScale> acl_scale;
};
}  // namespace ov::intel_cpu
