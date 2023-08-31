// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_convert.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;
using namespace InferenceEngine;

bool ACLConvertExecutor::init(const ConvertParams& convertParams,
                              const MemoryDescPtr& srcDesc,
                              const MemoryDescPtr& dstDesc,
                              const dnnl::primitive_attr& attr) {
    aclConvertParams = convertParams;

    auto srcPrecision = precisionToAclDataType(aclConvertParams.srcPrc);
    auto dstPrecision = precisionToAclDataType(aclConvertParams.dstPrc);
    isCopyOp = aclConvertParams.srcPrc == aclConvertParams.dstPrc;
    // NECast does not support S8. It could be replaced with QASYMM8_SIGNED
    if (!isCopyOp && srcPrecision == DataType::S8) {
        srcPrecision = DataType::QASYMM8_SIGNED;
    }
    if (!isCopyOp && dstPrecision == DataType::S8) {
        dstPrecision = DataType::QASYMM8_SIGNED;
    }
    auto srcDims = srcDesc->getShape().getStaticDims();
    auto dstDims = dstDesc->getShape().getStaticDims();
    auto srcDataLayout = getAclDataLayoutByMemoryDesc(srcDesc);
    auto dstDataLayout = getAclDataLayoutByMemoryDesc(dstDesc);
    auto srcTensorInfo = TensorInfo(shapeCast(collapse_dims_to_max_rank(srcDims)), 1, srcPrecision, srcDataLayout);
    auto dstTensorInfo = TensorInfo(shapeCast(collapse_dims_to_max_rank(dstDims)), 1, dstPrecision, dstDataLayout);
    if (isCopyOp) {
        Status s = NECopy::validate(&srcTensorInfo, &dstTensorInfo);
        if (!s) {
            DEBUG_LOG("NECopy validation failed: ", s.error_description());
            return false;
        }
    } else {
        Status s = NECast::validate(&srcTensorInfo, &dstTensorInfo, ConvertPolicy::SATURATE);
        if (!s) {
            DEBUG_LOG("NECast validation failed: ", s.error_description());
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    if (isCopyOp) {
        acl_copy = std::make_unique<NECopy>();
        acl_copy->configure(&srcTensor, &dstTensor);
    } else {
        acl_cast = std::make_unique<NECast>();
        acl_cast->configure(&srcTensor, &dstTensor, ConvertPolicy::SATURATE);
    }
    return true;
}

void ACLConvertExecutor::exec(const MemoryCPtr& src, const MemoryPtr& dst) {
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

bool ACLConvertExecutorBuilder::isSupported(const ConvertParams& convertParams,
                                            const MemoryDescPtr& srcDesc,
                                            const MemoryDescPtr& dstDesc) const {
    if (convertParams.srcPrc != convertParams.dstPrc) {
        if (!one_of(convertParams.srcPrc,
                    Precision::I8,
                    Precision::U8,
                    Precision::U16,
                    Precision::I16,
                    Precision::FP16,
                    Precision::I32,
                    Precision::FP32)) {
            DEBUG_LOG("NECopy does not support source precision: ", convertParams.srcPrc.name());
            return false;
        }
        if ((convertParams.srcPrc == Precision::I8 && !one_of(convertParams.dstPrc,
                                                              Precision::I16,
                                                              Precision::I32,
                                                              Precision::FP16,
                                                              Precision::FP32)) ||
            (convertParams.srcPrc == Precision::U8 && !one_of(convertParams.dstPrc,
                                                              Precision::U16,
                                                              Precision::I16,
                                                              Precision::I32,
                                                              Precision::FP16,
                                                              Precision::FP32)) ||
            (convertParams.srcPrc == Precision::U16 && !one_of(convertParams.dstPrc,
                                                               Precision::U8,
                                                               Precision::U32)) ||
            (convertParams.srcPrc == Precision::I16 && !one_of(convertParams.dstPrc,
                                                               Precision::I8,
                                                               Precision::U8,
                                                               Precision::I32)) ||
            (convertParams.srcPrc == Precision::FP16 && !one_of(convertParams.dstPrc,
                                                                Precision::I8,
                                                                Precision::FP32,
                                                                Precision::I32,
                                                                Precision::U8)) ||
            (convertParams.srcPrc == Precision::I32 && !one_of(convertParams.dstPrc,
                                                               Precision::I8,
                                                               Precision::FP16,
                                                               Precision::FP32,
                                                               Precision::U8)) ||
            (convertParams.srcPrc == Precision::FP32 && !one_of(convertParams.dstPrc,
                                                                Precision::BF16,
                                                                Precision::FP16,
                                                                Precision::I32))) {
            DEBUG_LOG("NECopy does not support passed combination of source and destination precisions. ",
                      "source precision: ", convertParams.srcPrc.name(), " destination precsion: ", convertParams.dstPrc.name());
            return false;
        }
    }
    return true;
}

} // namespace intel_cpu
} // namespace ov