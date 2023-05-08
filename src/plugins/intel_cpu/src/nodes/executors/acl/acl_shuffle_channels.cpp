// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_shuffle_channels.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

ACLShuffleChannelsExecutor::ACLShuffleChannelsExecutor(const ExecutorContext::CPtr context) : ShuffleChannelsExecutor(context) {}

bool ACLShuffleChannelsExecutor::init(const ShuffleChannelsAttributes &shuffleChannelsAttributes,
                                      const std::vector<MemoryDescPtr> &srcDescs,
                                      const std::vector<MemoryDescPtr> &dstDescs,
                                      const dnnl::primitive_attr &attr) {
    aclShuffleChannelsAttributes = shuffleChannelsAttributes;

    if (aclShuffleChannelsAttributes.axis != 1) {
        return false;
    }

    auto srcDims = srcDescs[0]->getShape().getDims();
    auto dstDims = dstDescs[0]->getShape().getDims();

    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);

    auto srcPrecision = precisionToAclDataType(srcDescs[0]->getPrecision());
    auto dstPrecision = precisionToAclDataType(dstDescs[0]->getPrecision());

    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1, dstPrecision, dstDataLayout);
    arm_compute::ActivationLayerInfo activationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::IDENTITY);

    // Bad accuracy for axis = 1 and group = 3
    if (aclShuffleChannelsAttributes.axis == 1 && aclShuffleChannelsAttributes.group == 3) { return false; }

    if (aclShuffleChannelsAttributes.group == 1) {
        if (!arm_compute::NEActivationLayer::validate(&srcTensorInfo, &dstTensorInfo, activationLayerInfo)) {
            return false;
        }
    } else {
        if (!arm_compute::NEChannelShuffleLayer::validate(&srcTensorInfo, &dstTensorInfo, aclShuffleChannelsAttributes.group)) {
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (aclShuffleChannelsAttributes.group == 1) {
        aclActivationLayer = std::make_unique<arm_compute::NEActivationLayer>();
        aclActivationLayer->configure(&srcTensor, &dstTensor, activationLayerInfo);
    } else {
        aclChannelShuffleLayer = std::make_unique<arm_compute::NEChannelShuffleLayer>();
        aclChannelShuffleLayer->configure(&srcTensor, &dstTensor, aclShuffleChannelsAttributes.group);
    }
    return true;
}

void ACLShuffleChannelsExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    if (aclShuffleChannelsAttributes.group == 1) {
        aclActivationLayer->run();
    } else {
        aclChannelShuffleLayer->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov