// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_lowp_fullyconnected.hpp"

#include "acl_fullyconnected_utils.hpp"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/acl/acl_utils.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

static bool checkPostOps(const PostOps& postOps) {
    if (postOps.empty()) {
        return true;
    }

    if (postOps.size() != 1) {
        return false;
    }

    const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
    return checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()));
}

static void initFCAttrs(const FCAttrs& attrs,
                        ACLTensorAttrs& aclTensorAttrs,
                        ACLFCAttrs& aclfcAttrs,
                        const MemoryArgs& memory,
                        arm_compute::GEMMInfo& fullyConnectedLayerInfo,
                        const PostOps& postOps) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (!postOps.empty()) {
        auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
        fullyConnectedLayerInfo.set_activation_info(
            getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                   activation->alpha(),
                                   activation->beta(),
                                   activation->gamma()));
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLLowpFullyConnectedExecutor::ACLLowpFullyConnectedExecutor(const FCAttrs& attrs,
                                                             const PostOps& postOps,
                                                             const MemoryArgs& memory,
                                                             const ExecutorContext::CPtr& context) {
    dequantizationScales = getDeQuantizedScales(memory);
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, gemmInfo, postOps);
    packedWeights = acl_fc_executor::prepareWeightMemory(memory,
                                                         context,
                                                         attrs,
                                                         aclfcAttrs,
                                                         postOps,
                                                         expectedWeightFormat,
                                                         weiTensorInfo);
}

bool ACLLowpFullyConnectedExecutor::supports(const FCConfig& config) {
    const auto src0 = srcType(config);
    const auto src1 = weiType(config);
    const auto dst = dstType(config);
    if ((src0 != ov::element::i8) || (src1 != ov::element::i8) || (dst != ov::element::f32)) {
        return false;
    }

    VERIFY(checkPostOps(config.postOps), UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U, 4U), UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLLowpFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    acl_fc_executor::updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLLowpFullyConnectedExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    auto& tensor_info = aclMemoryInfos[ACLArgs::ACL_SRC_0];
    if (dequantizationScales.empty()) {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(1.f));
    } else {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(dequantizationScales[0]));
    }

    auto& tensor_info_weights = aclMemoryInfos[ACLArgs::ACL_WEI];
    tensor_info_weights->set_quantization_info(arm_compute::QuantizationInfo(1.f));

    auto matMulValid = arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                                           aclMemoryInfos[ACLArgs::ACL_WEI].get(),
                                                                           aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
                                                                           aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                                           gemmInfo);
    return matMulValid;
}

ACLFunction ACLLowpFullyConnectedExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto gemm = std::make_unique<arm_compute::NEGEMMLowpMatrixMultiplyCore>();
    gemm->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                    aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                    aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                    aclMemoryTensors.at(ACLArgs::ACL_DST).get(),
                    gemmInfo);

    if (aclfcAttrs.isConvertedWeights || !aclfcAttrs.weightsNonTransposed) {
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_WEI] = false;
        aclMemoryTensors[ACLArgs::ACL_WEI]->allocator()->import_memory(packedWeights->getData());
    }
    return gemm;
}

std::shared_ptr<arm_compute::TensorInfo> ACLLowpFullyConnectedExecutor::initTensorInfo(
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
