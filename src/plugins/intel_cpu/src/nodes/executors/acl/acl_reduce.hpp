// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../reduce.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class AclReduceExecutor : public ReduceExecutor {
public:
    AclReduceExecutor(const ExecutorContext::CPtr context);

    bool init(const ReduceAttrs& reduceAttrs,
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
    std::function<void()> exec_func;
    ReduceAttrs reduceAttrs;
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Coordinates axesMean;
    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
};

class AclReduceExecutorBuilder : public ReduceExecutorBuilder {
public:
    bool isSupported(const ReduceAttrs& reduceAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (reduceAttrs.operation == Algorithm::ReduceMean) {
            if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
               (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
                srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16)) {
                DEBUG_LOG("NEReduceMean does not support precisions:",
                        " src[0]=", srcDescs[0]->getPrecision(),
                        " dst[0]=", dstDescs[0]->getPrecision());
                return false;
            }
        } else {
            if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
               (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
                srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16 &&
                srcDescs[0]->getPrecision() != InferenceEngine::Precision::I32)) {
                DEBUG_LOG("NEReductionOperation does not support precisions:",
                        " src[0]=", srcDescs[0]->getPrecision(),
                        " dst[0]=", dstDescs[0]->getPrecision());
                return false;
            }
        }
        return true;
    }

    ReduceExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclReduceExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov