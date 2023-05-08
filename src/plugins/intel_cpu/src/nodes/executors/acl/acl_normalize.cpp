// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_normalize.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

ACLNormalizeL2Executor::ACLNormalizeL2Executor(const ExecutorContext::CPtr context) : NormalizeL2Executor(context) {}

bool ACLNormalizeL2Executor::init(const NormalizeL2Attrs &normalizeL2Attrs, const std::vector<MemoryDescPtr> &srcDescs,
                             const std::vector<MemoryDescPtr> &dstDescs, const dnnl::primitive_attr &attr) {
    aclNormalizeL2Attrs = normalizeL2Attrs;
    auto srcDims = srcDescs[0]->getShape().getDims();
    auto dstDims = dstDescs[0]->getShape().getDims();

    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);

    auto srcPrecision = precisionToAclDataType(srcDescs[0]->getPrecision());
    auto dstPrecision = precisionToAclDataType(dstDescs[0]->getPrecision());

    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1, dstPrecision, dstDataLayout);

    auto&& axes = aclNormalizeL2Attrs.axisSet;
    int axis = axisCast(*axes.begin(), srcDescs[0]->getShape().getRank());
    //  Maximum supported actual reduction axis : 2
    if (axes.size() != 1 || axis > 2) return false;

    if (!arm_compute::NEL2NormalizeLayer::validate(&srcTensorInfo, &dstTensorInfo, axis, aclNormalizeL2Attrs.eps)) {
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    aclNEL2NormalizeLayer = std::make_unique<arm_compute::NEL2NormalizeLayer>();
    aclNEL2NormalizeLayer->configure(&srcTensor, &dstTensor, axis, aclNormalizeL2Attrs.eps);
    return true;
}

void ACLNormalizeL2Executor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                  const void **post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    aclNEL2NormalizeLayer->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov