// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/deconv.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class AclDeconvExecutor : public DeconvExecutor {
public:
    AclDeconvExecutor();

    bool init(const DeconvAttrs& deconvAttrs,
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
    DeconvAttrs deconvAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor weiTensor;
    arm_compute::Tensor biasTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEDeconvolutionLayer> deconv = nullptr;
};

class AclDeconvExecutorBuilder : public DeconvExecutorBuilder {
public:
    bool isSupported(const DeconvAttrs& deconvAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if ((srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
             srcDescs[1]->getPrecision() != InferenceEngine::Precision::FP32 &&
             dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32) &&
            (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16 &&
             srcDescs[1]->getPrecision() != InferenceEngine::Precision::FP16 &&
             dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP16)) {
            DEBUG_LOG("AclDeconvExecutor does not support precisions:",
                      " src[0]=", srcDescs[0]->getPrecision(),
                      " src[1]=", srcDescs[1]->getPrecision(),
                      " dst[0]=", dstDescs[0]->getPrecision());
            return false;
        }
        if (deconvAttrs.withBiases &&
           srcDescs[2]->getPrecision() != srcDescs[0]->getPrecision()) {
            DEBUG_LOG("AclDeconvExecutor does not support precisions:",
                      " src[2]=", srcDescs[2]->getPrecision());
            return false;
           }

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("AclDeconvExecutor does not support layouts:",
                    " src[0]=", srcDescs[0]->serializeFormat(),
                    " src[1]=", srcDescs[1]->serializeFormat(),
                    " dst=", dstDescs[0]->serializeFormat());
                return false;
              }
        if (deconvAttrs.withBiases &&
            !(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[1]->hasLayoutType(LayoutType::ncsp) &&
              srcDescs[2]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[1]->hasLayoutType(LayoutType::nspc) &&
              srcDescs[2]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("AclDeconvExecutor does not support layouts:",
                    " src[0]=", srcDescs[0]->serializeFormat(),
                    " src[1]=", srcDescs[1]->serializeFormat(),
                    " src[2]=", srcDescs[2]->serializeFormat(),
                    " dst=", dstDescs[0]->serializeFormat());
                return false;
              }
        return true;
    }

    DeconvExecutorPtr makeExecutor() const override {
        return std::make_shared<AclDeconvExecutor>();
    }
};

}   // namespace intel_cpu
}   // namespace ov