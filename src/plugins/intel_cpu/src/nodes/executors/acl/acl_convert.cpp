// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_convert.hpp"
#include "acl_utils.hpp"

ov::intel_cpu::ACLConvertExecutor::ACLConvertExecutor(const ov::intel_cpu::ExecutorContext::CPtr context)
        : ConvertExecutor(context) {}

bool ov::intel_cpu::ACLConvertExecutor::init(const ov::intel_cpu::ConvertParams &convertParams,
                                             const std::vector<MemoryDescPtr> &srcDescs,
                                             const std::vector<MemoryDescPtr> &dstDescs,
                                             const dnnl::primitive_attr &attr) {
    aclConvertParams = convertParams;
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);
    if (srcDataLayout == arm_compute::DataLayout::UNKNOWN &&
        dstDataLayout == arm_compute::DataLayout::UNKNOWN)  {
        return false;
    }

    auto srcPrecision = precisionToAclDataType(aclConvertParams.srcPrc);
    auto dstPrecision = precisionToAclDataType(aclConvertParams.dstPrc);
    if (srcPrecision == arm_compute::DataType::F32 && dstPrecision == arm_compute::DataType::U8) {
        return false;
    }

    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1, dstPrecision, dstDataLayout);
    if (aclConvertParams.srcPrc == aclConvertParams.dstPrc) {
        if (!arm_compute::NECopy::validate(&srcTensorInfo, &dstTensorInfo)) {
            return false;
        }
    } else {
        if (!arm_compute::NECast::validate(&srcTensorInfo, &dstTensorInfo, arm_compute::ConvertPolicy::SATURATE)) {
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (aclConvertParams.srcPrc == aclConvertParams.dstPrc) {
        acl_copy = std::make_unique<arm_compute::NECopy>();
        acl_copy->configure(&srcTensor, &dstTensor);
    } else {
        acl_cast = std::make_unique<arm_compute::NECast>();
        acl_cast->configure(&srcTensor, &dstTensor, arm_compute::ConvertPolicy::SATURATE);
    }
    return true;
}

void ov::intel_cpu::ACLConvertExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    if (aclConvertParams.srcPrc == aclConvertParams.dstPrc) {
        acl_copy->run();
    } else {
        acl_cast->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}
