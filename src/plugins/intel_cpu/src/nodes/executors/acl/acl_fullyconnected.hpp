// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

class ACLFullyConnectedExecutor : public ACLCommonExecutor {
public:
    ACLFullyConnectedExecutor(const FCAttrs& attrs,
                  const PostOps& postOps,
                  const MemoryArgs& memory,
                  const ExecutorContext::CPtr context);

    static bool supports(const FCConfig& config);

    void updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLMemoryInfos & aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLMemoryTensors & aclMemoryTensors) override;

    impl_desc_type implType() const override {
        return impl_desc_type::gemm_acl;
    }
private:
    arm_compute::FullyConnectedLayerInfo fullyConnectedLayerInfo;
    arm_compute::WeightsInfo weightsInfo;
};

using ACLFullyConnectedExecutorPtr = std::shared_ptr<ACLFullyConnectedExecutor>;

}  // namespace intel_cpu
}  // namespace ov
