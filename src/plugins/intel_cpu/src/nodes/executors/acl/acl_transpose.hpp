// Copyright (C) 2018-2026 Intel Corporation
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
    ACLTransposeExecutor(const TransposeParams& transposeParams,
                         const MemoryDescPtr& srcDesc,
                         const MemoryDescPtr& dstDesc,
                         ExecutorContext::CPtr context);
    static bool supports(const TransposeConfig& config);
    static ExecutorPtr create(const TransposeAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

private:
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEPermute> acl_permute;
};

}  // namespace ov::intel_cpu
