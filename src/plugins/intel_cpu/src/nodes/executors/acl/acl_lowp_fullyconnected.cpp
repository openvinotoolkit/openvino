// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_lowp_fullyconnected.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/QuantizationInfo.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/utils/quantization/AsymmHelpers.h>
#include <arm_compute/function_info/GEMMInfo.h>

#include <any>
#include <memory>

#include "acl_fullyconnected_utils.hpp"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/acl/acl_utils.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// ACL destination requantization supports only per-tensor scale/shift, so the fused FakeQuantize
// must carry single-element input scale/shift.
static bool isPerTensorFakeQuantize(const FakeQuantizePostOp& fq) {
    return fq.inputScale().size() == 1 && fq.inputShift().size() <= 1;
}

static bool checkPostOps(const PostOps& postOps) {
    if (postOps.empty()) {
        return true;
    }

    if (postOps.size() != 1) {
        return false;
    }

    if (const auto* const activation = std::any_cast<const ActivationPostOp>(postOps.data())) {
        return checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()));
    }
    if (const auto* const fq = std::any_cast<const FakeQuantizePostOp>(postOps.data())) {
        // Per-tensor FakeQuantize is fused into the GEMM as a quantized output stage (i8/u8 dst).
        return isPerTensorFakeQuantize(*fq);
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
        if (const auto* const activation = std::any_cast<const ActivationPostOp>(attrs.postOps.data())) {
            fullyConnectedLayerInfo.set_activation_info(
                getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                       activation->alpha(),
                                       activation->beta(),
                                       activation->gamma()));
        }
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLLowpFullyConnectedExecutor::ACLLowpFullyConnectedExecutor(const FCAttrs& attrs,
                                                             const MemoryArgs& memory,
                                                             const ExecutorContext::CPtr& context) {
    dequantizationScales = attrs.dqScales;
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, gemmInfo);

    hasQuantizedDst = any_of(memory.at(ARG_DST)->getPrecision(), ov::element::i8, ov::element::u8);
    if (hasQuantizedDst && !attrs.postOps.empty()) {
        if (const auto* const fq = std::any_cast<const FakeQuantizePostOp>(attrs.postOps.data())) {
            fqInputScale = fq->inputScale();
            fqInputShift = fq->inputShift();
            foldFakeQuantizeOutputShift(fqInputShift, fq->outputScale(), fq->outputShift());
        }
    }

    packedWeights =
        acl_fc_executor::prepareWeightMemory(memory, context, attrs, aclfcAttrs, expectedWeightFormat, weiTensorInfo);
}

bool ACLLowpFullyConnectedExecutor::supports(const FCConfig& config) {
    VERIFY(aclSupported({config.descs.at(ARG_SRC), config.descs.at(ARG_WEI), config.descs.at(ARG_DST)}),
           UNSUPPORTED_ACL_COMMON_PRECONDITION);
    VERIFY(any_of(srcType(config), ov::element::u8, ov::element::i8), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(weiType(config) == ov::element::i8, UNSUPPORTED_WEI_PRECISIONS);
    VERIFY(any_of(dstType(config), ov::element::f32, ov::element::i8, ov::element::u8), UNSUPPORTED_DST_PRECISIONS);
    VERIFY(checkPostOps(config.attrs.postOps), UNSUPPORTED_TYPE_OF_POSTOPS);
    const bool isQuantizedDst = any_of(dstType(config), ov::element::i8, ov::element::u8);
    if (isQuantizedDst) {
        const bool hasFakeQuantize = !config.attrs.postOps.empty() &&
                                     std::any_cast<const FakeQuantizePostOp>(config.attrs.postOps.data()) != nullptr;
        VERIFY(hasFakeQuantize, UNSUPPORTED_TYPE_OF_POSTOPS);
        if (!config.descs.at(ARG_BIAS)->empty()) {
            VERIFY(config.descs.at(ARG_BIAS)->getPrecision() == ov::element::i32, UNSUPPORTED_BIAS_PRECISIONS);
        }
    }
    VERIFY(any_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(any_of(weiRank(config), 2U, 3U, 4U), UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLLowpFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    acl_fc_executor::updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLLowpFullyConnectedExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    const auto& tensor_info = aclMemoryInfos[ACLArgs::ACL_SRC_0];
    tensor_info->set_quantization_info(arm_compute::QuantizationInfo(1.F));

    const auto& tensor_info_weights = aclMemoryInfos[ACLArgs::ACL_WEI];
    tensor_info_weights->set_quantization_info(dequantizationScales.empty()
                                                   ? arm_compute::QuantizationInfo(1.F)
                                                   : arm_compute::QuantizationInfo(dequantizationScales));

    const auto& tensor_info_dst = aclMemoryInfos[ACLArgs::ACL_DST];
    if (hasQuantizedDst) {
        const bool quantizedDst =
            any_of(tensor_info_dst->data_type(), arm_compute::DataType::QASYMM8, arm_compute::DataType::QASYMM8_SIGNED);
        OPENVINO_ASSERT(quantizedDst, "ACLLowpFullyConnectedExecutor: quantized dst expected for the output stage");
        const auto dstPrecision =
            tensor_info_dst->data_type() == arm_compute::DataType::QASYMM8_SIGNED ? ov::element::i8 : ov::element::u8;
        tensor_info_dst->set_quantization_info(getDstQuantizationInfo(fqInputScale, fqInputShift, dstPrecision));

        const auto oqinfo = tensor_info_dst->quantization_info().uniform();
        const auto [type_min, type_max] =
            arm_compute::quantization::get_min_max_values_from_quantized_data_type(tensor_info_dst->data_type());

        arm_compute::GEMMLowpOutputStageInfo outputStage;
        outputStage.type = arm_compute::GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        outputStage.gemmlowp_offset = oqinfo.offset;
        outputStage.gemmlowp_min_bound = type_min;
        outputStage.gemmlowp_max_bound = type_max;
        outputStage.is_quantized_per_channel = dequantizationScales.size() > 1;
        outputStage.output_data_type = tensor_info_dst->data_type();
        const auto multipliersStatus =
            arm_compute::quantization::calculate_quantized_multipliers(tensor_info->quantization_info(),
                                                                       tensor_info_weights->quantization_info(),
                                                                       tensor_info_dst->quantization_info(),
                                                                       outputStage);
        if (!multipliersStatus) {
            return multipliersStatus;
        }
        gemmInfo.set_gemmlowp_output_stage(outputStage);
    }

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
    return ACLCommonExecutor::initTensorInfo(tensorShape, convertToQuantizedType(dataType), dataLayout);
}

}  // namespace ov::intel_cpu
