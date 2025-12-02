// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/function_info/FullyConnectedLayerInfo.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>

#include <any>
#include <memory>

#include "acl_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/acl/acl_fullyconnected_utils.hpp"
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
    if (postOps.size() > 1) {
        return false;
    }
    if (const auto* const activation = std::any_cast<ActivationPostOp>(postOps.data())) {
        if (checkActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()))) {
            return true;
        }
    }
    return false;
}

static void initFCAttrs(const FCAttrs& attrs,
                        ACLTensorAttrs& aclTensorAttrs,
                        ACLFCAttrs& aclfcAttrs,
                        const MemoryArgs& memory,
                        arm_compute::FullyConnectedLayerInfo& fullyConnectedLayerInfo) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    aclfcAttrs.inputPrecision = memory.at(ARG_SRC)->getDescPtr()->getPrecision();
    fullyConnectedLayerInfo.transpose_weights = false;
    aclfcAttrs.weightsNonTransposed = attrs.weightsNonTransposed;

    if (!attrs.postOps.empty() && checkPostOps(attrs.postOps)) {
        const auto& activation = std::any_cast<const ActivationPostOp&>(attrs.postOps[0]);
        fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation.type()),
                                                                         activation.alpha(),
                                                                         activation.beta(),
                                                                         activation.gamma());
    }

    if (memory.at(ARG_SRC)->getPrecision() != memory.at(ARG_WEI)->getPrecision()) {
        aclfcAttrs.isConvertedWeights = true;
    }
}

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs& attrs,
                                                     const MemoryArgs& memory,
                                                     const ExecutorContext::CPtr& context) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo);
    packedWeights =
        acl_fc_executor::prepareWeightMemory(memory, context, attrs, aclfcAttrs, expectedWeightFormat, weiTensorInfo);
}

bool ACLFullyConnectedExecutor::supports(const FCConfig& config) {
    VERIFY(any_of(srcType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(any_of(weiType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_WEI_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2, UNSUPPORTED_NUMBER_OF_POSTOPS);

    VERIFY(checkPostOps(config.attrs.postOps), UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(any_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(any_of(weiRank(config), 2U, 3U), UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    acl_fc_executor::updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLFullyConnectedExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    if (aclfcAttrs.isConvertedWeights) {
        aclMemoryInfos[ACLArgs::ACL_WEI]->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_type());
    }
    int ic_total = aclMemoryInfos[ACLArgs::ACL_SRC_0]->dimension(0);
    return arm_compute::NEFullyConnectedLayer::validate(
        aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
        &weiTensorInfo,
        aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
        aclMemoryInfos[ACLArgs::ACL_DST].get(),
        fullyConnectedLayerInfo,
        expectedWeightFormat == arm_compute::WeightFormat::UNSPECIFIED
            ? arm_compute::WeightsInfo()
            : arm_compute::WeightsInfo(false, 1, 1, ic_total, false, expectedWeightFormat));
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    aclMemoryTensors[ACLArgs::ACL_WEI]->allocator()->init(weiTensorInfo);
    int icTotal = aclMemoryTensors[ACLArgs::ACL_WEI]->info()->dimension(0);
    neFC->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                    aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                    aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                    aclMemoryTensors[ACLArgs::ACL_DST].get(),
                    fullyConnectedLayerInfo,
                    expectedWeightFormat == arm_compute::WeightFormat::UNSPECIFIED
                        ? arm_compute::WeightsInfo()
                        : arm_compute::WeightsInfo(false, 1, 1, icTotal, false, expectedWeightFormat));
    // TODO: get rid of those flags and decide whether to import memory or not just based on input type
    if (aclfcAttrs.isWeightsRepacked || aclfcAttrs.isConvertedWeights) {
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_WEI] = false;
        aclMemoryTensors[ACLArgs::ACL_WEI]->allocator()->import_memory(packedWeights->getData());
    }
    return neFC;
}

}  // namespace ov::intel_cpu
