// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/convolution_config.hpp"

namespace ov::intel_cpu {

class ACLConvolutionExecutor : public ACLCommonExecutor {
public:
    ACLConvolutionExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;

protected:
    std::shared_ptr<arm_compute::TensorInfo> initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                                            const arm_compute::DataType& dataType,
                                                            const arm_compute::DataLayout& dataLayout) override;

private:
    ConvAttrs convAttrs;
    arm_compute::PadStrideInfo padStrideInfo;
    arm_compute::WeightsInfo weightsInfo;
    arm_compute::Size2D dilation;
    arm_compute::ActivationLayerInfo activationLayerInfo;
    bool enableFastMath = false;

    std::vector<float> inputScale;
    std::vector<float> outputScale;
    std::vector<float> inputShift;
    std::vector<float> outputShift;
    std::vector<float> weightScale;
};

using ACLConvolutionExecutorPtr = std::shared_ptr<ACLConvolutionExecutor>;

}  // namespace ov::intel_cpu
