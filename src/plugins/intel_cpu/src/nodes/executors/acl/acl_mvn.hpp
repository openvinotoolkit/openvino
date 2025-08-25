// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "acl_common_executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"

namespace ov::intel_cpu {

class ACLMVNExecutor : public ACLCommonExecutor {
public:
    ACLMVNExecutor(MVNAttrs attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& /*context*/)
        : aclMVNAtrrs(std::move(attrs)) {
        // Initialize ACL tensor attributes for src and dst tensors
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_SRC_0] = true;
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_DST] = true;

        // Determine layout type
        auto srcDesc = memory.at(ARG_SRC_0)->getDescPtr();
        isNHWCLayout = srcDesc->hasLayoutType(LayoutType::nspc);
        isNCHWLayout = srcDesc->hasLayoutType(LayoutType::ncsp);

        aclTensorAttrs.hasLayoutTypeNHWC = isNHWCLayout;

        // Don't collapse dimensions for MVN - we need the original shape info
        aclTensorAttrs.maxDimsShape = 100;  // Large value to prevent collapsing
    }

    static bool supports(const MVNConfig& config);

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

private:
    MVNAttrs aclMVNAtrrs;
    bool isNHWCLayout = false;
    bool isNCHWLayout = false;
};

using ACLMVNExecutorPtr = std::shared_ptr<ACLMVNExecutor>;
}  // namespace ov::intel_cpu
