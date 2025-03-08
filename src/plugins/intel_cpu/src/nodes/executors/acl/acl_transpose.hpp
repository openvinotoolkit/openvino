// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/Tensor.h"
#include "nodes/executors/transpose.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class ACLTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;

    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

private:
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEPermute> acl_permute;
};

class ACLTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const TransposeParams& transposeParams,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) && dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            DEBUG_LOG("NEPermute does not support layout:",
                      " src: ",
                      srcDescs[0]->serializeFormat(),
                      " dst: ",
                      dstDescs[0]->serializeFormat());
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
        return true;
    }

    [[nodiscard]] TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLTransposeExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
