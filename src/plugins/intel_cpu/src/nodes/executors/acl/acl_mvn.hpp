// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "acl_utils.hpp"
#include "nodes/executors/mvn.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class AclMVNExecutor : public MVNExecutor {
public:
    AclMVNExecutor(const ExecutorContext::CPtr context);

    bool init(const MVNAttrs& mvnAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEMeanStdDevNormalizationLayer> mvn = nullptr;
};

class AclMVNExecutorBuilder : public MVNExecutorBuilder {
public:
    bool isSupported(const MVNAttrs& mvnAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if ((srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
             srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16) ||
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
            getAclDataLayoutByMemoryDesc(srcDescs[0]) == arm_compute::DataLayout::NHWC) {
            DEBUG_LOG("initAcrossChannels = false is not supported by ACL for NHWC layout");
            return false;
        }

        return true;
    }

    MVNExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclMVNExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
