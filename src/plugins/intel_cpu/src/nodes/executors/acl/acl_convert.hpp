// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "nodes/executors/convert.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

class ACLConvertExecutor : public ConvertExecutor {
public:
    using ConvertExecutor::ConvertExecutor;
    bool init(const ConvertParams& convertParams,
              const MemoryDescPtr& srcDesc,
              const MemoryDescPtr& dstDesc,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::acl;
    };

protected:
    ConvertParams aclConvertParams;
    bool isCopyOp;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NECopy> acl_copy;
    std::unique_ptr<arm_compute::NECast> acl_cast;
};

class ACLConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const ConvertParams& convertParams,
                                   const MemoryDescPtr& srcDesc,
                                   const MemoryDescPtr& dstDesc) const override;
    [[nodiscard]] ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLConvertExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
