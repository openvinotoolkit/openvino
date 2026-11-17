// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/Tensor.h>

#include <cstdint>
#include <memory>

#include "acl_common_executor.hpp"
#include "nodes/executors/convolution_config.hpp"

namespace ov::intel_cpu {

class ACLConvolutionExecutor : public ACLCommonExecutor {
public:
    ACLConvolutionExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;
    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

private:
    ConvAttrs convAttrs;
    arm_compute::PadStrideInfo padStrideInfo;
    arm_compute::WeightsInfo weightsInfo;
    arm_compute::Size2D dilation;
    arm_compute::ActivationLayerInfo activationLayerInfo;

    std::vector<float> fqInputScale;
    std::vector<float> fqOutputScale;
    std::vector<float> fqInputShift;
    std::vector<float> fqOutputShift;
    std::vector<float> weightScale;
    // Per-tensor zero point of the int8 activations
    int32_t srcZeroPoint = 0;
    // ACL supports only a per-tensor weights scale for the int8 convolution with float output.
    // With per-channel scales, the scales, the bias and the activation are applied after ACL.
    bool applyScalesAfterAcl = false;
    arm_compute::ActivationLayerInfo postActivationInfo;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEActivationLayer> postActivation;
};

using ACLConvolutionExecutorPtr = std::shared_ptr<ACLConvolutionExecutor>;

}  // namespace ov::intel_cpu
