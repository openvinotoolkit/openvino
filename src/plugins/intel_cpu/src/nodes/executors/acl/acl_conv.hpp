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
    ~ACLConvolutionExecutor();

    static bool supports(const ConvConfig& config);

    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override;

    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;

    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;
    ACLFunction configureFunctionPostOp(const ACLTensors& aclMemoryTensors) override;

    arm_compute::TensorShape normalizeDimsTo2D(const arm_compute::TensorShape shape);

protected:
    std::shared_ptr<arm_compute::TensorInfo> initTensorInfo(const arm_compute::TensorShape& tensorShape,
                                                            const arm_compute::DataType& dataType,
                                                            const arm_compute::DataLayout& dataLayout) override;
private:
    std::shared_ptr<arm_compute::TensorInfo> dstTensorInfo;
    std::shared_ptr<arm_compute::Tensor> dstTensor;

    arm_compute::PadStrideInfo padStrideInfo;
    arm_compute::WeightsInfo weightsInfo;
    arm_compute::Size2D dilation;
    arm_compute::ActivationLayerInfo activationLayerInfo;
    bool enableFastMath;
    //unsigned int numGroups;
    ConvAttrs convAttrs;
    arm_compute::TensorInfo weiTensorInfo;
    std::vector<float> dequantizationScales;
    std::vector<float> inputScale, outputScale;
    std::vector<float> inputShift, outputShift;
    std::vector<float> weightScale;
};

using ACLConvolutionExecutorPtr = std::shared_ptr<ACLConvolutionExecutor>;

}  // namespace ov::intel_cpu
