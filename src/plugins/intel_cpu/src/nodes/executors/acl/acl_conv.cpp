// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv.hpp"

#include <common/primitive_desc_iface.hpp>
#include <cpu/acl/acl_utils.hpp>

#include "acl_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/reorder_prim.h"
#include "nodes/convert.h"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

ACLConvolutionExecutor::ACLConvolutionExecutor(const ConvAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context) {
    dequantizationScales = getDeQuantizedScales(memory);

    MemoryDescPtr srcMemPtr = memory.at(ARG_SRC_0)->getDescPtr();
    MemoryDescPtr weiMemPtr = memory.at(ARG_WEI)->getDescPtr();
    MemoryDescPtr dstMemPtr = memory.at(ARG_DST)->getDescPtr();
    
    Shape weiShape = weiMemPtr->getShape();
    Shape srcShape = srcMemPtr->getShape();
    Shape dstShape = dstMemPtr->getShape();

    size_t srcDims = srcShape.getRank();
    const int with_groups = weiShape.getRank() == srcDims + 1;

    const int kh = weiShape.getDims()[with_groups + srcDims - 2];
    const int kw = weiShape.getDims()[with_groups + srcDims - 1];

    const int oc = dstShape.getDims()[1];

    //TODO: disable fast math so far
    enableFastMath = false;

    weightsInfo = arm_compute::WeightsInfo(false, kw, kh, oc, false, arm_compute::WeightFormat::UNSPECIFIED);

    //TODO: how to get top and bottom paddings?
    padStrideInfo = arm_compute::PadStrideInfo(attrs.stride[0], attrs.stride[1], attrs.paddingL[0], attrs.paddingR[0]);

    dilation = arm_compute::Size2D(attrs.dilation[1] + 1, attrs.dilation[0] + 1);

    weightScale = attrs.dqScales;
    //TODO: do we need to check there is only 1 post op?
    if (!attrs.postOps.empty() && attrs.postOps.size() == 1) {
        if (const auto activation = std::any_cast<ActivationPostOp>(&attrs.postOps[0])) {
            activationLayerInfo = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                        activation->alpha(),
                                                        activation->beta(),
                                                        activation->gamma());
        } else if (const auto fq = std::any_cast<FakeQuantizePostOp>(&attrs.postOps[0])) {
            //char* sc = std::getenv("OV_CPU_SCALE");
            //float fl = std::stof(sc);
            inputScale = fq->inputScale();
            inputShift = fq->inputShift();
            //inputScale.push_back(std::stof(std::getenv("OV_CPU_D_SCALE")));//fq->inputScale();
            //inputShift.push_back(std::stof(std::getenv("OV_CPU_D_SHIFT")));//fq->inputShift();
            std::cout << "inputScale: " << inputScale[0] << "inputShift: " << inputShift[0] << std::endl;
            outputScale = fq->outputScale();
            outputShift = fq->outputShift();
        } else {
            DEBUG_LOG("ACLConvolutionExecutor: post op is not applied!");
        }

    }
}

arm_compute::TensorShape ACLConvolutionExecutor::normalizeDimsTo2D(const arm_compute::TensorShape shape) {
    size_t norm_dim = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    return arm_compute::TensorShape(shape[0], norm_dim);
}

void ACLConvolutionExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    //TODO: doing the same as for FC executor.
    //If this is correct than normalizeDimsTo2D/updateFCTensorsShapes should be moved from FC into common logic
    //aclMemoryShapes[ACLArgs::ACL_WEI] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_WEI]);
    //aclMemoryShapes[ACLArgs::ACL_SRC_0] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_SRC_0]);
    //aclMemoryShapes[ACLArgs::ACL_DST] = normalizeDimsTo2D(aclMemoryShapes[ACLArgs::ACL_DST]);
    //std::swap(aclMemoryShapes[ACLArgs::ACL_WEI][0], aclMemoryShapes[ACLArgs::ACL_WEI][1]);
}

bool ACLConvolutionExecutor::supports(const ConvConfig& config) {
    //std::cout << "ACLConvolutionExecutor::supports - PASSED PREC CHECKS!!!" << std::endl;
    return true;
}

arm_compute::Status ACLConvolutionExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    //auto& tensor_info = aclMemoryInfos[ACLArgs::ACL_SRC_0];
    //if (inputScale.empty()) {//(dequantizationScales.empty()) {
    //    aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0, 0, false));
    //} else {
        //tensor_info->set_quantization_info(arm_compute::QuantizationInfo(inputScale[0], inputShift[0], true/*dequantizationScales[0]*/));
        aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0, std::stof(std::getenv("OV_CPU_S_SHIFT"))/*1, 0*/, false));
    //}

    //auto& tensor_info_weights = aclMemoryInfos[ACLArgs::ACL_WEI];
    aclMemoryInfos[ACLArgs::ACL_WEI]->set_quantization_info(arm_compute::QuantizationInfo(weightScale.empty() ? 1.0 : weightScale[0], std::stof(std::getenv("OV_CPU_W_SHIFT"))/*1, 0*/, false));
    //auto& tensor_info_out = aclMemoryInfos[ACLArgs::ACL_DST];
    aclMemoryInfos[ACLArgs::ACL_DST]->set_quantization_info(arm_compute::QuantizationInfo(inputScale.empty() ? 1.0 : 1.0 / inputScale[0], std::stof(std::getenv("OV_CPU_D_SHIFT"))/*inputScale[0], inputShift[0]*//*1, 0*/, false));
    //tensor_info_out->set_quantization_info(arm_compute::QuantizationInfo(outputScale[0], outputShift[0], true));
    /*dstTensorInfo = std::make_shared<arm_compute::TensorInfo>(aclMemoryInfos[ACLArgs::ACL_DST].get()->tensor_shape(),
                                            aclMemoryInfos[ACLArgs::ACL_DST].get()->num_channels(),
                                            aclMemoryInfos[ACLArgs::ACL_SRC_0].get()->data_type(),
                                            arm_compute::QuantizationInfo(1, 0, true));*/
    //dstTensorInfo = std::make_shared<arm_compute::TensorInfo>(*aclMemoryInfos[ACLArgs::ACL_DST].get());
    //dstTensorInfo->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0].get()->data_type());
    //dstTensorInfo->set_quantization_info(arm_compute::QuantizationInfo(1, 0, true));//outputScale[0], -outputShift[0], true));//(1, 0, true));
    arm_compute::Status s = arm_compute::NEConvolutionLayer::validate(
        aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
        aclMemoryInfos[ACLArgs::ACL_WEI].get(),
        aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
        aclMemoryInfos[ACLArgs::ACL_DST].get(),//dstTensorInfo.get(),
        padStrideInfo,
        weightsInfo,
        dilation,
        activationLayerInfo,
        enableFastMath);

    return s;
}

/*ACLFunction ACLConvolutionExecutor::configureFunctionPostOp(const ACLTensors& aclMemoryTensors) {
    //FIXME: do a proper check whether dequantize is needed or not
    if (aclMemoryTensors[ACLArgs::ACL_DST].get()->info()->data_type() == dstTensor.get()->info()->data_type()) {
        return nullptr;
    }
    auto neDeq = std::make_unique<arm_compute::NEDequantizationLayer>();
    neDeq->configure(dstTensor.get(),
                     aclMemoryTensors[ACLArgs::ACL_DST].get());
    return neDeq;
}*/

ACLFunction ACLConvolutionExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto neConv = std::make_unique<arm_compute::NEConvolutionLayer>();
    //dstTensor = std::make_shared<arm_compute::Tensor>();
    //dstTensor->allocator()->init(*dstTensorInfo);

    neConv->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                      aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      //dstTensor.get(),
                      padStrideInfo,
                      weightsInfo,
                      dilation,
                      activationLayerInfo,
                      enableFastMath);
    //dstTensor->allocator()->allocate();
    return neConv;
}

/*ACLConvolutionExecutor::~ACLConvolutionExecutor() {
    dstTensor->allocator()->free();
}*/

std::shared_ptr<arm_compute::TensorInfo> ACLConvolutionExecutor::initTensorInfo(
    const arm_compute::TensorShape& tensorShape,
    const arm_compute::DataType& dataType,
    const arm_compute::DataLayout& dataLayout) {
    arm_compute::DataType result;
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
