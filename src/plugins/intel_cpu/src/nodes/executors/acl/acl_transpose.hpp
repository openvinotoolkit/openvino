// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

class ACLTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    static const impl_desc_type implType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEPermute> acl_permute;
};

class ACLTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            DEBUG_LOG("NEPermute does not support layout:",
                      " src: ", srcDescs[0]->serializeFormat(),
                      " dst: ", dstDescs[0]->serializeFormat());
            return false;
        }
        if (srcDescs[0]->getShape().getRank() > 4) {
            DEBUG_LOG("NEPermute supports up to 4D input tensor. Passed tensor rank: ",
                      srcDescs[0]->getShape().getRank());
            return false;
        }
        if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision()) {
            DEBUG_LOG("NEPermute requires the same input and output precisions");
            return false;
        }
        if (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
            srcDescs[0]->getPrecision() != InferenceEngine::Precision::I8) {
            DEBUG_LOG("NEPermute supports 1, 2, 4 bytes data types. FP16 implementation is disabled due to performance issues");
            return false;
        }
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov