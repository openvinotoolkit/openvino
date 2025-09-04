// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/eltwise_config.hpp"

namespace ov::intel_cpu {

class ACLEltwiseExecutor : public ACLCommonExecutor {
public:
    ACLEltwiseExecutor(const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const EltwiseConfig& config);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

private:
    EltwiseAttrs aclEltwiseAttrs;
};

using ACLEltwiseExecutorPtr = std::shared_ptr<ACLEltwiseExecutor>;

}  // namespace ov::intel_cpu