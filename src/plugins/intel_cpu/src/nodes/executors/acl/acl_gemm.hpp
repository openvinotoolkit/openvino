// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/gemm_config.hpp"

namespace ov {
namespace intel_cpu {

class ACLGEMMExecutor : public ACLCommonExecutor {
public:
    ACLGEMMExecutor(const GEMMAttrs& attrs,
                      const PostOps& postOps,
                      const MemoryArgs& memory,
                      const ExecutorContext::CPtr context);

    static bool supports(const GEMMConfig& config);

    void updateTensorsShapes(ACLMemoryShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLMemoryInfo & aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLMemoryTensors & aclMemoryTensors) override;

    impl_desc_type implType() const override {
        return impl_desc_type::gemm_acl;
    }

protected:
    ACLInfo initTensorInfo(const arm_compute::TensorShape& tensorShape,
                           const arm_compute::DataType& dataType,
                           const arm_compute::DataLayout& dataLayout) override;

private:
    arm_compute::FullyConnectedLayerInfo fullyConnectedLayerInfo;
    arm_compute::GEMMInfo gemmInfo;
    arm_compute::WeightsInfo weightsInfo;
};

using ACLGEMMExecutorPtr = std::shared_ptr<ACLGEMMExecutor>;

}  // namespace intel_cpu
}  // namespace ov
