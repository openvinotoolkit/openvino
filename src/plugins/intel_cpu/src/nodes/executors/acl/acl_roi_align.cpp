// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_roi_align.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclROIAlignExecutor::AclROIAlignExecutor(const ExecutorContext::CPtr context) : ROIAlignExecutor(context) {}

bool AclROIAlignExecutor::init(const ROIAlignAttrs& roialignAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    auto dstDims = dstDescs[0]->getShape().getDims();
    auto srcDims = srcDescs[0]->getShape().getDims();
    auto roiDims = srcDescs[1]->getShape().getDims();
    numRois = roiDims[0];
    //roi tensor shape is changed because ACL expects [N, 5] tensor while OV uses [N, 4] tensor
    roiDims[1] = 5;

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo roiTensorInfo = TensorInfo(shapeCast(roiDims), 1,
    precisionToAclDataType(srcDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[1]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    ROIPoolingLayerInfo poolInfo(roialignAttrs.pooledW, roialignAttrs.pooledH, roialignAttrs.spatialScale, roialignAttrs.samplingRatio);

    Status s = arm_compute::NEROIAlignLayer::validate(&srcTensorInfo, &roiTensorInfo, &dstTensorInfo, poolInfo);
    if (!s) {
        DEBUG_LOG("NEROIAlignLayer validation failed: ", s.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    roiTensor.allocator()->init(roiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    roialign = std::make_unique<arm_compute::NEROIAlignLayer>();
    roialign->configure(&srcTensor, &roiTensor, &dstTensor, poolInfo);

    return true;
}

void AclROIAlignExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    //roi tensor shape is changed because ACL expects [N, 5] tensor while OV uses [N, 4] tensor
    //the missing 5th dimension is batch id
    //roiBuffer presents [N, 5] roi vector that passed to ACL
    float* elem = reinterpret_cast<float*>(src[1]->GetPtr());
    int* batchId = reinterpret_cast<int*>(src[2]->GetPtr());
    for (int i = 0; i < numRois; i++) {
        roiBuffer.push_back(*(batchId + i));
        roiBuffer.push_back(*(elem + i * 4));
        roiBuffer.push_back(*(elem + i * 4 + 1));
        roiBuffer.push_back(*(elem + i * 4 + 2));
        roiBuffer.push_back(*(elem + i * 4 + 3));
    }

    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    roiTensor.allocator()->import_memory(roiBuffer.data());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    roialign->run();

    srcTensor.allocator()->free();
    roiTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov