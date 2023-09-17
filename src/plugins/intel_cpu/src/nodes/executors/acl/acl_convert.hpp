// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"
#include "utils/debug_capabilities.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLConvertExecutor : public ConvertExecutor {
public:
    using ConvertExecutor::ConvertExecutor;
    bool init(const ConvertParams& convertParams,
              const MemoryDescPtr& srcDesc,
              const MemoryDescPtr& dstDesc,
              const dnnl::primitive_attr &attr) override;
    void exec(const MemoryCPtr& src, const MemoryPtr& dst) override;
    impl_desc_type getImplType() const override { return implDescType; };
protected:
    ConvertParams aclConvertParams;
    bool isCopyOp;
    static const impl_desc_type implDescType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NECopy> acl_copy;
    std::unique_ptr<arm_compute::NECast> acl_cast;
};

class ACLConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    bool isSupported(const ConvertParams& convertParams,
                     const MemoryDescPtr& srcDesc,
                     const MemoryDescPtr& dstDesc) const override;
    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLConvertExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov