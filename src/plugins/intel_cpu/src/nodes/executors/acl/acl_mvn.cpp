// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_mvn.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclMVNExecutor::AclMVNExecutor(const ExecutorContext::CPtr context) : MVNExecutor(context) {}

bool AclMVNExecutor::init(const MVNAttrs& mvnAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    size_t X, Y;
    if (mvnAttrs.initAcrossChannels_) {
        if (srcDims.size() >= 2u) {
            Y = srcDims[0];
            X = srcDims[1];
            for (size_t i = 2; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else {
            Y = 1;
            X = srcDims[0];
        }
    } else {
        if (srcDims.size() > 2u) {
            Y = srcDims[0] * srcDims[1];
            X = srcDims[2];
            for (size_t i = 3; i < srcDims.size(); i++) {
                X *= srcDims[i];
            }
        } else if (srcDims.size() == 2u) {
            Y = srcDims[0] * srcDims[1];
            X = 1;
        } else {
            Y = srcDims[0];
            X = 1;
        }
    }

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(X, Y), 1, precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(X, Y), 1, precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));


    if (!arm_compute::NEMeanStdDevNormalizationLayer::validate(&srcTensorInfo, &dstTensorInfo, mvnAttrs.epsValue_))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    mvn = std::make_unique<arm_compute::NEMeanStdDevNormalizationLayer>();
    configureThreadSafe([&] { mvn->configure(&srcTensor, &dstTensor, mvnAttrs.epsValue_); });

    return true;
}

void AclMVNExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    mvn->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool AclMVNExecutorBuilder::isSupported(const MVNAttrs& mvnAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs) const {
        if ((srcDescs[0]->getPrecision() != ov::element::f32 &&
             srcDescs[0]->getPrecision() != ov::element::f16) ||
             srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision()) {
            DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support precisions:",
                      " src[0]=", srcDescs[0]->getPrecision(),
                      " dst[0]=", dstDescs[0]->getPrecision());
            return false;
        }

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support layout:",
                      " src: ", srcDescs[0]->serializeFormat(),
                      " dst: ", dstDescs[0]->serializeFormat());
            return false;
        }

        if (mvnAttrs.epsMode_ == MVNEpsMode::OUTSIDE_SQRT) {
            DEBUG_LOG("NEMeanStdDevNormalizationLayer does not support OUTSIDE_SQRT mode");
            return false;
        }
        if (!mvnAttrs.normalizeVariance_) {
            DEBUG_LOG("NEMeanStdDevNormalizationLayer supports normalize_variance=true only");
            return false;
        }
        if (!mvnAttrs.initAcrossChannels_ &&
            srcDescs[0]->hasLayoutType(LayoutType::nspc)) {
            DEBUG_LOG("initAcrossChannels = false is not supported by ACL for NHWC layout");
            return false;
        }

        return true;
    }

}   // namespace intel_cpu
}   // namespace ov
