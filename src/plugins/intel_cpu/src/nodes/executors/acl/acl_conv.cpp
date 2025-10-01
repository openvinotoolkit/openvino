// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
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
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "post_ops.hpp"

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

    const auto with_groups = static_cast<const int>(weiShape.getRank() == srcShape.getRank() + 1);
    const int kh = weiShape.getDims()[with_groups + srcShape.getRank() - 2];
    const int kw = weiShape.getDims()[with_groups + srcShape.getRank() - 1];
    const int oc = dstShape.getDims()[1];

    weightsInfo = arm_compute::WeightsInfo(false, kw, kh, oc, false, arm_compute::WeightFormat::UNSPECIFIED);
    auto paddingLeft = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[1] : attrs.paddingL[0];
    auto paddingRight = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[1] : attrs.paddingR[0];
    auto paddingTop = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[0] : 0;
    auto paddingBottom = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[0] : 0;
    padStrideInfo = arm_compute::PadStrideInfo(attrs.stride[0],
                                               attrs.stride[1],
                                               paddingLeft,
                                               paddingRight,
                                               paddingTop,
                                               paddingBottom,
                                               arm_compute::DimensionRoundingType::FLOOR);
    dilation = arm_compute::Size2D(attrs.dilation[1] + 1, attrs.dilation[0] + 1);

    if (attrs.postOps.size() == 1) {
        if (const auto* const activation = std::any_cast<ActivationPostOp>(attrs.postOps.data())) {
            activationLayerInfo = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                         activation->alpha(),
                                                         activation->beta(),
                                                         activation->gamma());
        } else if (const auto* const fq = std::any_cast<FakeQuantizePostOp>(attrs.postOps.data())) {
            fqInputScale = fq->inputScale();
            fqInputShift = fq->inputShift();
            fqOutputScale = fq->outputScale();
            fqOutputShift = fq->outputShift();
            if (fqOutputScale.size() == 1 && fqOutputScale[0] == 1.0F && fqOutputShift.size() == 1 &&
                fqOutputShift[0] == std::trunc(fqOutputShift[0])) {
                for (auto& v : fqInputShift) {
                    v += fqOutputShift[0];
                }
                fqOutputShift.clear();
            }
        } else {
            OPENVINO_THROW("ACLConvolutionExecutor: the executor supports FakeQuantize and Activation post ops only");
        }
    } else if (attrs.postOps.size() > 1) {
        OPENVINO_THROW("ACLConvolutionExecutor: ACL does not support more than 1 post op");
    }
}

bool ACLConvolutionExecutor::supports(const ConvConfig& config) {
    bool isQuantized = any_of(config.descs.at(ARG_SRC)->getPrecision(), ov::element::u8, ov::element::i8) &&
                       config.descs.at(ARG_WEI)->getPrecision() == ov::element::i8;

    VERIFY(isQuantized, UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(config.attrs.postOps.size() <= 1U, UNSUPPORTED_BY_EXECUTOR);

    return true;
}

arm_compute::Status ACLConvolutionExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    // Note: LPT propagate dequantization scales from src and weights on conv output, and the result scale
    // is applied as weight scale. So quantization configuration forms in the following way:
    // - src quantization info is always trivial
    // - weights: scale is equal to result dequantization scale after Convolution propagated by LPT
    //            shift is not supported
    // - destination: scale is formed based on requantization FakeQuantize parameters: scale = 1.0 / input scale
    //                shift = input shift
    aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0));
    aclMemoryInfos[ACLArgs::ACL_WEI]->set_quantization_info(
        arm_compute::QuantizationInfo(weightScale.empty() ? 1.0F : weightScale[0]));
    aclMemoryInfos[ACLArgs::ACL_DST]->set_quantization_info(
        arm_compute::QuantizationInfo(fqInputScale.empty() ? 1.0F : 1.0F / fqInputScale[0],
                                      fqInputShift.empty() ? 0 : fqInputShift[0]));

    return arm_compute::NEConvolutionLayer::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_WEI].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                     padStrideInfo,
                                                     weightsInfo,
                                                     dilation,
                                                     activationLayerInfo);
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
                      activationLayerInfo);
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
