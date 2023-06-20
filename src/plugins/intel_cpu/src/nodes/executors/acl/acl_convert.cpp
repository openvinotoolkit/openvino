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

    auto srcPrecision = precisionToAclDataType(aclConvertParams.srcPrc);
    auto dstPrecision = precisionToAclDataType(aclConvertParams.dstPrc);
    isCopyOp = aclConvertParams.srcPrc == aclConvertParams.dstPrc;
    //NECast does not support S8. It could be replaced with QASYMM8_SIGNED
    if (!isCopyOp && srcPrecision == arm_compute::DataType::S8) {
        srcPrecision = arm_compute::DataType::QASYMM8_SIGNED;
    }
    if (!isCopyOp && dstPrecision == arm_compute::DataType::S8) {
        dstPrecision = arm_compute::DataType::QASYMM8_SIGNED;
    }
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDescs[0]);
    auto srcTensorInfo = arm_compute::TensorInfo(shapeCast(srcDims), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = arm_compute::TensorInfo(shapeCast(dstDims), 1, dstPrecision, dstDataLayout);
    if (isCopyOp) {
        arm_compute::Status s = arm_compute::NECopy::validate(&srcTensorInfo, &dstTensorInfo);
        if (!s) {
            DEBUG_LOG("NECopy validation failed: ", s.error_description());
            return false;
        }
    } else {
        arm_compute::Status s = arm_compute::NECast::validate(&srcTensorInfo, &dstTensorInfo, arm_compute::ConvertPolicy::SATURATE);
        if (!s) {
            std::cout << "NECast validation failed: " <<  s.error_description() << std::endl;
            DEBUG_LOG("NECast validation failed: ", s.error_description());
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (isCopyOp) {
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

    if (isCopyOp) {
        acl_copy->run();
    } else {
        acl_cast->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}
