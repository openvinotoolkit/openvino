// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/QuantizationInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>

#include <any>
#include <cmath>
#include <memory>

#include "acl_utils.hpp"
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "post_ops.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

ACLConvolutionExecutor::ACLConvolutionExecutor(const ConvAttrs& attrs,
                                               const MemoryArgs& memory,
                                               [[maybe_unused]] const ExecutorContext::CPtr& context)
    : weightScale(attrs.dqScales) {
    MemoryDescPtr srcMemPtr = memory.at(ARG_SRC_0)->getDescPtr();
    MemoryDescPtr weiMemPtr = memory.at(ARG_WEI)->getDescPtr();
    MemoryDescPtr dstMemPtr = memory.at(ARG_DST)->getDescPtr();

    Shape weiShape = weiMemPtr->getShape();
    Shape srcShape = srcMemPtr->getShape();
    Shape dstShape = dstMemPtr->getShape();

    const int with_groups = static_cast<const int>(weiShape.getRank() == srcShape.getRank() + 1);
    const int kh = weiShape.getDims()[with_groups + srcShape.getRank() - 2];
    const int kw = weiShape.getDims()[with_groups + srcShape.getRank() - 1];
    const int oc = dstShape.getDims()[1];

    weightsInfo = arm_compute::WeightsInfo(false, kw, kh, oc, false, arm_compute::WeightFormat::UNSPECIFIED);
    padStrideInfo = arm_compute::PadStrideInfo(attrs.stride[0], attrs.stride[1], attrs.paddingL[0], attrs.paddingR[0]);
    dilation = arm_compute::Size2D(attrs.dilation[1] + 1, attrs.dilation[0] + 1);

    if (!attrs.postOps.empty() && attrs.postOps.size() == 1) {
        if (const auto* const activation = std::any_cast<ActivationPostOp>(attrs.postOps.data())) {
            activationLayerInfo = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                         activation->alpha(),
                                                         activation->beta(),
                                                         activation->gamma());
        } else if (const auto* const fq = std::any_cast<FakeQuantizePostOp>(attrs.postOps.data())) {
            inputScale = fq->inputScale();
            inputShift = fq->inputShift();
            outputScale = fq->outputScale();
            outputShift = fq->outputShift();
            if (outputScale.size() == 1 && outputScale[0] == 1.0F && outputShift.size() == 1 &&
                outputShift[0] == std::trunc(outputShift[0])) {
                for (auto& v : inputShift) {
                    v += outputShift[0];
                }
                outputShift.clear();
            }
        }
    } else {
        DEBUG_LOG("ACLConvolutionExecutor: post op is not applied!");
    }
}

arm_compute::Status ACLConvolutionExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0));
    aclMemoryInfos[ACLArgs::ACL_WEI]->set_quantization_info(
        arm_compute::QuantizationInfo(weightScale.empty() ? 1.0F : weightScale[0]));
    aclMemoryInfos[ACLArgs::ACL_DST]->set_quantization_info(
        arm_compute::QuantizationInfo(inputScale.empty() ? 1.0F : 1.0F / inputScale[0],
                                      inputShift.empty() ? 0 : inputShift[0],
                                      false));

    return arm_compute::NEConvolutionLayer::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_WEI].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                     padStrideInfo,
                                                     weightsInfo,
                                                     dilation,
                                                     activationLayerInfo,
                                                     enableFastMath);
}

ACLFunction ACLConvolutionExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto neConv = std::make_unique<arm_compute::NEConvolutionLayer>();

    neConv->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                      aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      padStrideInfo,
                      weightsInfo,
                      dilation,
                      activationLayerInfo,
                      enableFastMath);
    return neConv;
}

std::shared_ptr<arm_compute::TensorInfo> ACLConvolutionExecutor::initTensorInfo(
    const arm_compute::TensorShape& tensorShape,
    const arm_compute::DataType& dataType,
    const arm_compute::DataLayout& dataLayout) {
    arm_compute::DataType result = arm_compute::DataType::UNKNOWN;
    switch (dataType) {
    case arm_compute::DataType::S8: {
        result = arm_compute::DataType::QASYMM8_SIGNED;
        break;
    }
    case arm_compute::DataType::U8: {
        result = arm_compute::DataType::QASYMM8;
        break;
    }
    default: {
        result = dataType;
        break;
    }
    }

    return ACLCommonExecutor::initTensorInfo(tensorShape, result, dataLayout);
}

}  // namespace ov::intel_cpu
