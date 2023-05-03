// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/space_to_depth.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLSpaceToDepthExecutor : public SpaceToDepthExecutor {
public:
    explicit ACLSpaceToDepthExecutor(const ExecutorContext::CPtr context);
    bool init(const SpaceToDepthAttrs& spaceToDepthAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implDescType; }
    ~ACLSpaceToDepthExecutor() override = default;
private:
    SpaceToDepthAttrs aclSpaceToDepthAttrs;
    impl_desc_type implDescType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NESpaceToDepthLayer> acl_space_to_depth;
};

class ACLSpaceToDepthExecutorBuilder : public SpaceToDepthExecutorBuilder{
public:
    ~ACLSpaceToDepthExecutorBuilder() = default;

    bool isSupported(const SpaceToDepthAttrs& spaceToDepthAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (spaceToDepthAttrs.mode != SpaceToDepthAttrs::Mode::BLOCKS_FIRST) { return false; }
        return true;
    };

    SpaceToDepthExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLSpaceToDepthExecutor>(context);
    };
};

}   // namespace intel_cpu
}   // namespace ov