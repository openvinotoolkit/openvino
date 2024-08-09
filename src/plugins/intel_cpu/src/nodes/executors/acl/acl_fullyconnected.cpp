// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_fullyconnected.hpp"
#include "acl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"

namespace ov {
namespace intel_cpu {

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs,
                                                     const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) {
    initFCAttrs(attrs, aclTensorAttrs, aclfcAttrs, memory, fullyConnectedLayerInfo, postOps);
    packedWeights = prepareWeightMemory(memory, context, attrs, aclfcAttrs, postOps);
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    arm_compute::FullyConnectedLayerInfo tmpFullyConnectedLayerInfo;
    VERIFY(one_of(srcType(config), ov::element::f16, ov::element::f32),     UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(one_of(weiType(config), ov::element::f16, ov::element::f32),     UNSUPPORTED_WEI_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2,                                      UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(checkAndInitPostOps(config.postOps, tmpFullyConnectedLayerInfo), UNSUPPORTED_TYPE_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U),                             UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U),                                 UNSUPPORTED_WEI_RANK);
    return true;
}

void ACLFullyConnectedExecutor::updateTensorsShapes(ACLShapes& aclMemoryShapes) {
    updateFCTensorsShapes(aclMemoryShapes);
}

arm_compute::Status ACLFullyConnectedExecutor::validateTensorsInfo(const ACLInfos & aclMemoryInfos) {
    if (aclfcAttrs.isConvertedWeights) {
        aclMemoryInfos[ACLArgs::ACL_WEI]->set_data_type(aclMemoryInfos[ACLArgs::ACL_SRC_0]->data_type());
    }
    return arm_compute::NEFullyConnectedLayer::validate(
            aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
            aclMemoryInfos[ACLArgs::ACL_WEI].get(),
            aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
            aclMemoryInfos[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLTensors & aclMemoryTensors) {
    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    neFC->configure(
            aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
            aclMemoryTensors[ACLArgs::ACL_WEI].get(),
            aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
            aclMemoryTensors[ACLArgs::ACL_DST].get(),
            fullyConnectedLayerInfo,
            weightsInfo);

    if (aclfcAttrs.isConvertedWeights || !aclfcAttrs.weightsNonTransposed) {
        aclTensorAttrs.memoryUsageIndicator[ACLArgs::ACL_WEI] = false;
        aclMemoryTensors[ACLArgs::ACL_WEI]->allocator()->import_memory(packedWeights->getData());
    }
    return neFC;
}

}   // namespace intel_cpu
}   // namespace ov
