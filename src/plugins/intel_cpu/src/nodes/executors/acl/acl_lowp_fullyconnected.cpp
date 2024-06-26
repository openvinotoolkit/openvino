// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_lowp_fullyconnected.hpp"

#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "nodes/executors/acl/acl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "acl_weights.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

static bool checkPostOps(const PostOps &postOps) {
    if (postOps.empty()) {
        return true;
    }

    if (postOps.size() != 1) {
        return false;
    }

    const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
    return checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()));
}

void initFCAttrs(const FCAttrs &attrs,
                 ACLTensorAttrs& aclTensorAttrs,
                 ACLFCAttrs& aclfcAttrs,
                 const MemoryArgs &memory,
                 arm_compute::GEMMInfo& gemmInfo,
                 const PostOps &postOps) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    // TODO: not completed
    //fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    // TODO: not completed
    //fullyConnectedLayerInfo.transpose_weights = false;
    gemmInfo.set_pretranspose_A(false);
    gemmInfo.set_pretranspose_B(false);
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (!postOps.empty()) {
        auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0]);
        fullyConnectedLayerInfo.set_activation_info(getActivationLayerInfo(
                convertToEltwiseAlgorithm(activation->type()),
                activation->alpha(), activation->beta(), activation->gamma()));
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLLowpFullyConnectedExecutor::ACLLowpFullyConnectedExecutor(const FCAttrs &attrs,
                                                             const PostOps &postOps,
                                                             const MemoryArgs &memory,
                                                             const ExecutorContext::CPtr& context) : dequantizationScales(attrs.dequantizationScales) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, gemmInfo, postOps);
    packedWeights = prepareWeightMemory(memory, context, attrs, aclfcAttrs, postOps);
}

bool ACLLowpFullyConnectedExecutor::supports(const FCConfig &config) {
    const auto src0 = srcType(config);
    const auto src1 = weiType(config);
    const auto dst = dstType(config);
    if ((src0 != ov::element::i8) || (src1 != ov::element::i8) || (dst != ov::element::f32)) {
        return false;
    }

    VERIFY(checkPostOps(config.postOps), UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U, 4U), UNSUPPORTED_WEI_RANK);
    VERIFY(static_cast<FCAttrs>(config.attrs).dequantizationScales.size() <= 1, UNSUPPORTED_PER_CHANNEL_QUANTIZATION);
    return true;
}

void ACLLowpFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLLowpFullyConnectedExecutor::validateTensorsInfo(const ACLInfos & aclMemoryInfos) {
    // TODO: debug only
    //const auto src0 = aclMemoryInfos[ACLArgs::ACL_SRC_0].get();
    //const auto src1 = aclMemoryInfos[ACLArgs::ACL_WEI].get();
    //const auto dst = aclMemoryInfos[ACLArgs::ACL_DST].get();

    auto &tensor_info = aclMemoryInfos[ACLArgs::ACL_SRC_0];
    if (dequantizationScales.empty()) {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(1.f));
    } else {
        tensor_info->set_quantization_info(arm_compute::QuantizationInfo(dequantizationScales[0]));
    }

    auto& tensor_info_weights = aclMemoryInfos[ACLArgs::ACL_WEI];
    tensor_info_weights->set_quantization_info(arm_compute::QuantizationInfo(1.f));

    const auto matMulValid = arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            nullptr, //aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            gemmInfo);
    return matMulValid;
}

ACLFunction ACLLowpFullyConnectedExecutor::configureFunction(const ACLTensors & aclMemoryTensors) {
    auto gemm = std::make_unique<arm_compute::NEGEMMLowpMatrixMultiplyCore>();
    gemm->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
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

// TODO: move to ACLLowpExecutor
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

}   // namespace intel_cpu
}   // namespace ov
