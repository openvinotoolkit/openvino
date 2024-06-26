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

namespace ov {
namespace intel_cpu {

ACLFullyConnectedExecutor::ACLFullyConnectedExecutor(const FCAttrs &attrs, const PostOps &postOps,
                                                     const MemoryArgs &memory,
                                                     const ExecutorContext::CPtr context) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);
    fullyConnectedLayerInfo.weights_trained_layout = getAclDataLayoutByMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    fullyConnectedLayerInfo.transpose_weights = !attrs.weightsNonTransposed;

    // Add postops
    if (!postOps.empty() && postOps.size() == 1) {
        if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOps[0])) {
            fullyConnectedLayerInfo.activation_info = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                                             activation->alpha(),
                                                                             activation->beta(),
                                                                             activation->gamma());
        }
    }
}

bool ACLFullyConnectedExecutor::supports(const FCConfig &config) {
    VERIFY(one_of(srcType(config), ov::element::f16, ov::element::f32), UNSUPPORTED_SRC_PRECISIONS);
    VERIFY(postOpsNumbers(config) < 2,          UNSUPPORTED_NUMBER_OF_POSTOPS);
    VERIFY(one_of(srcRank(config), 2U, 3U, 4U), UNSUPPORTED_SRC_RANK);
    VERIFY(one_of(weiRank(config), 2U, 3U),     UNSUPPORTED_WEI_RANK);
    return true;
}

arm_compute::Status ACLFullyConnectedExecutor::updateTensorsInfo(const ACLMemoryMap& acl_memory) {
    auto wei_shape = acl_memory.at(ARG_WEI)->info()->tensor_shape();
    if (wei_shape.num_dimensions() == 3U) {
        acl_memory.at(ARG_WEI)->info()->set_tensor_shape({wei_shape[0] * wei_shape[1], wei_shape[2]});
    }

    auto src_shape = acl_memory.at(ARG_SRC)->info()->tensor_shape();
    if (one_of(src_shape.num_dimensions(), 3U, 4U)) {
        acl_memory.at(ARG_SRC)->info()->set_tensor_shape({
            acl_memory.at(ARG_WEI)->info()->tensor_shape()[0],
            src_shape.total_size() / acl_memory.at(ARG_WEI)->info()->tensor_shape()[0]});
    }

    if (one_of(acl_memory.at(ARG_DST)->info()->tensor_shape().num_dimensions(), 3U, 4U)) {
        acl_memory.at(ARG_DST)->info()->set_tensor_shape({
            acl_memory.at(ARG_WEI)->info()->tensor_shape()[1],
            acl_memory.at(ARG_SRC)->info()->tensor_shape()[1]});
    }

    if (!fullyConnectedLayerInfo.transpose_weights) {
        arm_compute::TensorShape temp_weights_shape = acl_memory.at(ARG_WEI)->info()->tensor_shape();
        std::swap(temp_weights_shape[0], temp_weights_shape[1]);
        acl_memory.at(ARG_WEI)->info()->set_tensor_shape(temp_weights_shape);
    }

    return arm_compute::NEFullyConnectedLayer::validate(
            getACLInfo(acl_memory.at(ARG_SRC)),
            getACLInfo(acl_memory.at(ARG_WEI)),
            getACLInfo(acl_memory.at(ARG_BIAS)),
            getACLInfo(acl_memory.at(ARG_DST)),
            fullyConnectedLayerInfo,
            weightsInfo);
}

ACLFunction ACLFullyConnectedExecutor::configureFunction(const ACLMemoryMap& acl_memory) {
    auto neFC = std::make_unique<arm_compute::NEFullyConnectedLayer>();
    neFC->configure(
            acl_memory.at(ARG_SRC).get(),
            acl_memory.at(ARG_WEI).get(),
            acl_memory.at(ARG_BIAS).get(),
            acl_memory.at(ARG_DST).get(),
            fullyConnectedLayerInfo,
            weightsInfo);
    return neFC;
}

}   // namespace intel_cpu
}   // namespace ov
