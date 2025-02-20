// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "acl_fullyconnected_utils.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov::intel_cpu {

class ACLFullyConnectedExecutor : public ACLCommonExecutor {
public:
    ACLFullyConnectedExecutor(const FCAttrs& attrs,
                              const PostOps& postOps,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    static bool supports(const FCConfig& config);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

private:
    arm_compute::FullyConnectedLayerInfo fullyConnectedLayerInfo;
    arm_compute::WeightFormat expectedWeightFormat;
    MemoryCPtr packedWeights;
    ACLFCAttrs aclfcAttrs;
    arm_compute::TensorInfo weiTensorInfo;
};

using ACLFullyConnectedExecutorPtr = std::shared_ptr<ACLFullyConnectedExecutor>;

}  // namespace ov::intel_cpu
