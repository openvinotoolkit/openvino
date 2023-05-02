// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_transpose.hpp"
#include "acl_utils.hpp"

std::size_t AxisCast(const std::size_t axis, const std::size_t shapeSize) {
    return shapeSize - axis - 1;
}


ov::intel_cpu::ACLTransposeExecutor::ACLTransposeExecutor(const ExecutorContext::CPtr context) : TransposeExecutor(context) {}

bool ov::intel_cpu::ACLTransposeExecutor::init(const ov::intel_cpu::TransposeParams &transposeParams,
                                               const std::vector<MemoryDescPtr> &srcDescs,
                                               const std::vector<MemoryDescPtr> &dstDescs,
                                               const dnnl::primitive_attr &attr) {
    if (transposeParams.transposeExecution != TransposeParams::NOT_REF &&
        srcDescs[0]->getShape().getRank() > 4) { return false; }

    auto inputOrder = transposeParams.permuteParams.order;
    if (inputOrder.empty()) {
        inputOrder.resize(srcDescs[0]->getShape().getRank());
        std::iota(inputOrder.begin(), inputOrder.end(), 0);
        std::reverse(inputOrder.begin(), inputOrder.end());
    }
    arm_compute::PermutationVector order;
    const auto maxSupportedNumOfDimensions = (inputOrder.size() < 4) ? 3u : 4u;
    for (unsigned int i = 0; i < maxSupportedNumOfDimensions; ++i) {
        order.set(i, i);
    }
    for (size_t i = 0; i < inputOrder.size(); ++i) {
        order.set(i, AxisCast(inputOrder[AxisCast(i, inputOrder.size())], inputOrder.size()));
    }
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1,
                                                 precisionToAclDataType(srcDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(srcDescs[0]));
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1,
                                                 precisionToAclDataType(dstDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(dstDescs[0]));
    if (!arm_compute::NEPermute::validate(&srcTensorInfo, &dstTensorInfo, order)) {
        return false;
    }
    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    acl_permute = std::make_unique<arm_compute::NEPermute>();
    acl_permute->configure(&srcTensor, &dstTensor, order);
    return true;
}

void ov::intel_cpu::ACLTransposeExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst,
                                               const int MB) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    acl_permute->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}
