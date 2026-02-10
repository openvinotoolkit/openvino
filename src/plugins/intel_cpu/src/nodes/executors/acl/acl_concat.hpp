// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include <arm_compute/runtime/Tensor.h>

#include "nodes/executors/concat.hpp"
#include "nodes/executors/executor.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class AclConcatExecutor : public ConcatExecutor {
public:
    using ConcatExecutor::ConcatExecutor;
    bool init(const ConcatAttrs& concatAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type getImplType() const override {
        return impl_desc_type::acl;
    }

private:
    std::vector<arm_compute::Tensor> srcTensors;
    arm_compute::Tensor dstTensor;
    arm_compute::NEConcatenateLayer concatLayer;
};

class AclConcatExecutorBuilder : public ConcatExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const ConcatAttrs& concatAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;
    [[nodiscard]] ConcatExecutorPtr makeExecutor(ExecutorContext::CPtr context) const override {
        return std::make_shared<AclConcatExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
