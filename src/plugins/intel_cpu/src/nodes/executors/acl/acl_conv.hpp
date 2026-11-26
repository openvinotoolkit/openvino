// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_common_executor.hpp"
#include "nodes/executors/convolution_config.hpp"

namespace ov::intel_cpu {

class ACLConvolutionExecutor : public ACLCommonExecutor {
public:
    ACLConvolutionExecutor(const ConvAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConvConfig& config);
#if defined(OPENVINO_ARCH_ARM64)
    static bool supportsOc1JitGeometry(const VectorDims& src,
                                       const VectorDims& weights,
                                       const VectorDims& dst,
                                       const std::vector<size_t>& strides,
                                       const std::vector<size_t>& dilations,
                                       const std::vector<ptrdiff_t>& padsBegin,
                                       const std::vector<ptrdiff_t>& padsEnd);
#endif
    void updateTensorsShapes(ACLShapes& aclMemoryShapes) override {}
    arm_compute::Status validateTensorsInfo(const ACLInfos& aclMemoryInfos) override;
    ACLFunction configureFunction(const ACLTensors& aclMemoryTensors) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return useOc1Jit ? impl_desc_type::jit_asimd : impl_desc_type::acl;
    }

private:
    std::shared_ptr<CpuParallel> cpuParallel;
    bool useOc1Jit = false;
    ConvAttrs convAttrs;
    arm_compute::PadStrideInfo padStrideInfo;
    arm_compute::WeightsInfo weightsInfo;
    arm_compute::Size2D dilation;
    arm_compute::ActivationLayerInfo activationLayerInfo;

    std::vector<float> fqInputScale;
    std::vector<float> fqOutputScale;
    std::vector<float> fqInputShift;
    std::vector<float> fqOutputShift;
    int32_t srcZeroPoint = 0;
    std::vector<float> weightScale;
};

using ACLConvolutionExecutorPtr = std::shared_ptr<ACLConvolutionExecutor>;

}  // namespace ov::intel_cpu
