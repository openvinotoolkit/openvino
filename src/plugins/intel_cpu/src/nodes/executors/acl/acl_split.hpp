// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/runtime/NEON/NEFunctions.h>

#include <memory>
#include <vector>

#include "acl_utils.hpp"
#include "nodes/executors/split.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class AclSplitExecutor : public SplitExecutor {
public:
    explicit AclSplitExecutor(ExecutorContext::CPtr context);

    bool init(const SplitAttrs& splitAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;
    std::unique_ptr<arm_compute::IFunction> acl_split;
    arm_compute::Tensor srcTensor;
    std::vector<arm_compute::Tensor> dstTensors;
    unsigned int aclAxis = 0;
};

class AclSplitExecutorBuilder : public SplitExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const SplitAttrs& splitAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] SplitExecutorPtr makeExecutor(ExecutorContext::CPtr context) const override {
        return std::make_shared<AclSplitExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
