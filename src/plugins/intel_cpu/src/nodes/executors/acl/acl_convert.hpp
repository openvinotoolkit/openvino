// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLConvertExecutor : public ConvertExecutor {
public:
    explicit ACLConvertExecutor(const ExecutorContext::CPtr context);
    bool init(const ConvertParams& convertParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    impl_desc_type getImplType() const override { return implDescType; };
    ~ACLConvertExecutor() override = default;
protected:
    ConvertParams aclConvertParams;
    impl_desc_type implDescType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NECopy> acl_copy;
    std::unique_ptr<arm_compute::NECast> acl_cast;
};


class ACLConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    ~ACLConvertExecutorBuilder() = default;
    bool isSupported(const ConvertParams& convertParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }
    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLConvertExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov