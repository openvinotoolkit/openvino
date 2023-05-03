// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "acl_space_to_depth.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

ACLSpaceToDepthExecutor::ACLSpaceToDepthExecutor(const ExecutorContext::CPtr context) : SpaceToDepthExecutor(context) {}

bool ACLSpaceToDepthExecutor::init(const SpaceToDepthAttrs &spaceToDepthAttrs,
                                      const std::vector<MemoryDescPtr> &srcDescs,
                                      const std::vector<MemoryDescPtr> &dstDescs,
                                      const dnnl::primitive_attr &attr) {
    aclSpaceToDepthAttrs = spaceToDepthAttrs;
    auto srcDims = srcDescs[0]->getShape().getDims();
    auto dstDims = dstDescs[0]->getShape().getDims();

    if (srcDims.size() > 4) { return false; }

    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);

    if (srcDataLayout == arm_compute::DataLayout::UNKNOWN || dstDataLayout == arm_compute::DataLayout::UNKNOWN) { return false; }

    auto srcPrecision = precisionToAclDataType(srcDescs[0]->getPrecision());
    auto dstPrecision = precisionToAclDataType(dstDescs[0]->getPrecision());

    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1, dstPrecision, dstDataLayout);

    if (!arm_compute::NESpaceToDepthLayer::validate(&srcTensorInfo, &dstTensorInfo, aclSpaceToDepthAttrs.blockSize)) {
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);
    acl_space_to_depth = std::make_unique<arm_compute::NESpaceToDepthLayer>();
    acl_space_to_depth->configure(&srcTensor, &dstTensor, aclSpaceToDepthAttrs.blockSize);
    return true;
}

void ACLSpaceToDepthExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    acl_space_to_depth->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov