// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/mvn_config.hpp"

namespace ov {
namespace intel_cpu {

class ACLMVNExecutor : public ACLCommonExecutor {
public:
    ACLMVNExecutor(const MVNAttrs& attrs,
                   const PostOps& postOps,
                   const MemoryArgs& memory,
                   const ExecutorContext::CPtr context) : aclMVNAtrrs(attrs) {}

    static bool supports(const MVNConfig& config);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos & aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors & aclMemoryTensors) override;

private:
    MVNAttrs aclMVNAtrrs;
};

using ACLMVNExecutorPtr = std::shared_ptr<ACLMVNExecutor>;
}   // namespace intel_cpu
}   // namespace ov