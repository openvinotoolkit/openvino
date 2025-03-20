// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "acl_fullyconnected_utils.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov::intel_cpu {

class ACLLowpFullyConnectedExecutor : public ACLCommonExecutor {
public:
    ACLLowpFullyConnectedExecutor(const FCAttrs& attrs,
                                  const PostOps& postOps,
                                  const MemoryArgs& memory,
                                  const ExecutorContext::CPtr& context);

    static bool supports(const FCConfig& config);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::gemm_acl;
    }

protected:
    std::shared_ptr<arm_compute::TensorInfo> initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                                            const arm_compute::DataType& dataType,
                                                            const arm_compute::DataLayout& dataLayout) override;

private:
    arm_compute::GEMMInfo gemmInfo;
    arm_compute::WeightFormat expectedWeightFormat;
    arm_compute::TensorInfo weiTensorInfo;

    MemoryCPtr packedWeights;
    ACLFCAttrs aclfcAttrs;
    std::vector<float> dequantizationScales;
};

using ACLLowpFullyConnectedExecutorPtr = std::shared_ptr<ACLLowpFullyConnectedExecutor>;

}  // namespace ov::intel_cpu
