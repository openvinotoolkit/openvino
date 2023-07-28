// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_convert.hpp"
#include "acl_utils.hpp"

bool ov::intel_cpu::ACLConvertExecutor::init(const ov::intel_cpu::ConvertParams& convertParams,
                                             const MemoryDescPtr& srcDesc,
                                             const MemoryDescPtr& dstDesc,
                                             const dnnl::primitive_attr& attr) {
    aclConvertParams = convertParams;

    auto srcPrecision = precisionToAclDataType(aclConvertParams.srcPrc);
    auto dstPrecision = precisionToAclDataType(aclConvertParams.dstPrc);
    isCopyOp = aclConvertParams.srcPrc == aclConvertParams.dstPrc;
    // NECast does not support S8. It could be replaced with QASYMM8_SIGNED
    if (!isCopyOp && srcPrecision == arm_compute::DataType::S8) {
        srcPrecision = arm_compute::DataType::QASYMM8_SIGNED;
    }
    if (!isCopyOp && dstPrecision == arm_compute::DataType::S8) {
        dstPrecision = arm_compute::DataType::QASYMM8_SIGNED;
    }
    auto srcDims = srcDesc->getShape().getStaticDims();
    auto dstDims = dstDesc->getShape().getStaticDims();
    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDesc);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDesc);
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

void ov::intel_cpu::ACLConvertExecutor::exec(const MemoryCPtr& src, const MemoryPtr& dst) {
    srcTensor.allocator()->import_memory(src->getData());
    dstTensor.allocator()->import_memory(dst->getData());

    if (isCopyOp) {
        acl_copy->run();
    } else {
        acl_cast->run();
    }

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool ov::intel_cpu::ACLConvertExecutorBuilder::isSupported(const ConvertParams& convertParams,
                                                           const MemoryDescPtr& srcDesc,
                                                           const MemoryDescPtr& dstDesc) const {
    if (convertParams.srcPrc != convertParams.dstPrc) {
        if (!one_of(convertParams.srcPrc,
                    InferenceEngine::Precision::I8,
                    InferenceEngine::Precision::U8,
                    InferenceEngine::Precision::U16,
                    InferenceEngine::Precision::I16,
                    InferenceEngine::Precision::FP16,
                    InferenceEngine::Precision::I32,
                    InferenceEngine::Precision::FP32)) {
            DEBUG_LOG("NECopy does not support source precision: ", convertParams.srcPrc.name());
            return false;
        }
        if ((convertParams.srcPrc == InferenceEngine::Precision::I8 && !one_of(convertParams.dstPrc,
                                                                               InferenceEngine::Precision::I16,
                                                                               InferenceEngine::Precision::I32,
                                                                               InferenceEngine::Precision::FP16,
                                                                               InferenceEngine::Precision::FP32)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::U8 && !one_of(convertParams.dstPrc,
                                                                               InferenceEngine::Precision::U16,
                                                                               InferenceEngine::Precision::I16,
                                                                               InferenceEngine::Precision::I32,
                                                                               InferenceEngine::Precision::FP16,
                                                                               InferenceEngine::Precision::FP32)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::U16 && !one_of(convertParams.dstPrc,
                                                                                InferenceEngine::Precision::U8,
                                                                                InferenceEngine::Precision::U32)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::I16 && !one_of(convertParams.dstPrc,
                                                                                InferenceEngine::Precision::I8,
                                                                                InferenceEngine::Precision::U8,
                                                                                InferenceEngine::Precision::I32)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::FP16 && !one_of(convertParams.dstPrc,
                                                                                 InferenceEngine::Precision::I8,
                                                                                 InferenceEngine::Precision::FP32,
                                                                                 InferenceEngine::Precision::I32,
                                                                                 InferenceEngine::Precision::U8)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::I32 && !one_of(convertParams.dstPrc,
                                                                                InferenceEngine::Precision::I8,
                                                                                InferenceEngine::Precision::FP16,
                                                                                InferenceEngine::Precision::FP32,
                                                                                InferenceEngine::Precision::U8)) ||
            (convertParams.srcPrc == InferenceEngine::Precision::FP32 && !one_of(convertParams.dstPrc,
                                                                                 InferenceEngine::Precision::BF16,
                                                                                 InferenceEngine::Precision::FP16,
                                                                                 InferenceEngine::Precision::I32))) {
            DEBUG_LOG("NECopy does not support passed combination of source and destination precisions. ",
                      "source precision: ", convertParams.srcPrc.name(), " destination precsion: ", convertParams.dstPrc.name());
            return false;
        }
    }
    return true;
}