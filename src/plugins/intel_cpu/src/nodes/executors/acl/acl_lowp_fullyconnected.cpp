// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_lowp_fullyconnected.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/QuantizationInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/function_info/GEMMInfo.h>

#include <any>
#include <memory>

#include "acl_fullyconnected_utils.hpp"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/acl/acl_utils.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

static bool checkPostOps(const PostOps& postOps) {
    if (postOps.empty()) {
        return true;
    }

    if (postOps.size() != 1) {
        return false;
    }

    if (const auto& activation = std::any_cast<const ActivationPostOp>(postOps.data())) {
        return checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()));
    }
    return false;
}

static void initFCAttrs(const FCAttrs& attrs,
                        ACLTensorAttrs& aclTensorAttrs,
                        ACLFCAttrs& aclfcAttrs,
                        const MemoryArgs& memory,
                        arm_compute::GEMMInfo& fullyConnectedLayerInfo) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (!attrs.postOps.empty()) {
        const auto& activation = std::any_cast<const ActivationPostOp&>(attrs.postOps[0]);
        fullyConnectedLayerInfo.set_activation_info(getActivationLayerInfo(convertToEltwiseAlgorithm(activation.type()),
                                                                           activation.alpha(),
                                                                           activation.beta(),
                                                                           activation.gamma()));
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLLowpFullyConnectedExecutor::ACLLowpFullyConnectedExecutor(const FCAttrs& attrs,
                                                             const MemoryArgs& memory,
                                                             const ExecutorContext::CPtr& context) {
    dequantizationScales = getDeQuantizedScales(memory);
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, gemmInfo);
    packedWeights =
        acl_fc_executor::prepareWeightMemory(memory, context, attrs, aclfcAttrs, expectedWeightFormat, weiTensorInfo);
}

bool ACLLowpFullyConnectedExecutor::supports(const FCConfig& config) {
    const auto src0 = srcType(config);
    const auto src1 = weiType(config);
    const auto dst = dstType(config);
    if ((src0 != ov::element::i8) || (src1 != ov::element::i8) || (dst != ov::element::f32)) {
        return false;
    }

    VERIFY(checkPostOps(config.attrs.postOps), UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(any_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(any_of(weiRank(config), 2U, 3U, 4U), UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLLowpFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    acl_fc_executor::updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLLowpFullyConnectedExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    const auto& tensor_info = aclMemoryInfos[ACLArgs::ACL_SRC_0];
    if (dequantizationScales.empty()) {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(1.F));
    } else {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(dequantizationScales[0]));
    }

    const auto& tensor_info_weights = aclMemoryInfos[ACLArgs::ACL_WEI];
    tensor_info_weights->set_quantization_info(arm_compute::QuantizationInfo(1.F));

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
    const auto result = [&]() {
        switch (dataType) {
        case arm_compute::DataType::S8:
            return arm_compute::DataType::QASYMM8_SIGNED;
        case arm_compute::DataType::U8:
            return arm_compute::DataType::QASYMM8;
        default:
            return dataType;
        }
    }();

    return ACLCommonExecutor::initTensorInfo(tensorShape, result, dataLayout);
}

}  // namespace ov::intel_cpu
