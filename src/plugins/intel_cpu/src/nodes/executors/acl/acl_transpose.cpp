// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_transpose.hpp"

#include "acl_utils.hpp"

bool ov::intel_cpu::ACLTransposeExecutor::init(const ov::intel_cpu::TransposeParams& transposeParams,
                                               const std::vector<MemoryDescPtr>& srcDescs,
                                               const std::vector<MemoryDescPtr>& dstDescs,
                                               const dnnl::primitive_attr& attr) {
    auto inputOrder = transposeParams.permuteParams.order;
    if (inputOrder.empty()) {
        inputOrder.resize(srcDescs[0]->getShape().getRank());
        std::iota(inputOrder.begin(), inputOrder.end(), 0);
    }

    std::vector<int> vec;
    if (srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
        auto changeLayoutToNhwc = [](VectorDims shape) -> VectorDims {
            std::swap(shape[1], shape[2]);
            std::swap(shape[2], shape[3]);
            return shape;
        };
        auto srcDims = changeLayoutToNhwc(srcDescs[0]->getShape().getStaticDims());
        auto dstDims = changeLayoutToNhwc(dstDescs[0]->getShape().getStaticDims());
        for (int i = inputOrder.size() - 1; i >= 0; --i) {
            auto it = find(srcDims.rbegin(), srcDims.rend(), dstDims[i]);
            int index = it - srcDims.rbegin();
            vec.push_back(index);
        }
    } else {
        for (unsigned int i = 0; i < inputOrder.size(); ++i) {
            vec.push_back(axisCast(inputOrder[i], inputOrder.size()));
        }
        std::reverse(vec.begin(), vec.end());
    }
    arm_compute::PermutationVector order;
    for (unsigned int i = 0; i < inputOrder.size(); ++i) {
        order.set(i, vec[i]);
    }

    auto srcDims = shapeCast(srcDescs[0]->getShape().getDims());
    auto dstDims = shapeCast(dstDescs[0]->getShape().getDims());

    if (srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&srcDims, &dstDims});
    }
    auto srcTensorInfo = arm_compute::TensorInfo(srcDims,
                                                 1,
                                                 precisionToAclDataType(srcDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(srcDescs[0]));
    auto dstTensorInfo = arm_compute::TensorInfo(dstDims,
                                                 1,
                                                 precisionToAclDataType(dstDescs[0]->getPrecision()),
                                                 getAclDataLayoutByMemoryDesc(dstDescs[0]));
    arm_compute::Status status = arm_compute::NEPermute::validate(&srcTensorInfo, &dstTensorInfo, order);
    if (!status) {
        DEBUG_LOG("NEPermute validation failed: ", status.error_description());
        return false;
    }
    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    acl_permute = std::make_unique<arm_compute::NEPermute>();
    configureThreadSafe([&] {
        acl_permute->configure(&srcTensor, &dstTensor, order);
    });
    return true;
}

void ov::intel_cpu::ACLTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    acl_permute->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}
