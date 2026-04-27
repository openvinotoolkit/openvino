// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "nodes/executors/pad.hpp"

namespace ov::intel_cpu {

class AclPadExecutor : public PadExecutor {
public:
    AclPadExecutor(ExecutorContext::CPtr context);

    bool init(const PadAttrs& padAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void* post_ops_data_) override;

    [[nodiscard]] impl_desc_type getImplType() const override {
        return implType;
    }

private:
    impl_desc_type implType = impl_desc_type::acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    std::unique_ptr<arm_compute::NEPadLayer> pad;
};

class AclPadExecutorBuilder : public PadExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const PadAttrs& padAttrs,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override;

    [[nodiscard]] PadExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclPadExecutor>(context);
    }
};

}  // namespace ov::intel_cpu