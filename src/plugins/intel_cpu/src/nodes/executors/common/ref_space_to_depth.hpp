// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/space_to_depth.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov {
namespace intel_cpu {

class CommonSpaceToDepthExecutor : public SpaceToDepthExecutor {
public:
    explicit CommonSpaceToDepthExecutor(const ExecutorContext::CPtr context);
    bool init(const SpaceToDepthAttrs& spaceToDepthAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return commonSpaceToDepthAttrs.implDescType; }
    ~CommonSpaceToDepthExecutor() override = default;
private:
    SpaceToDepthAttrs commonSpaceToDepthAttrs;
    std::unique_ptr<PermuteKernel> permuteKernel;
};

class CommonSpaceToDepthExecutorBuilder : public SpaceToDepthExecutorBuilder{
public:
    ~CommonSpaceToDepthExecutorBuilder() = default;
    bool isSupported(const SpaceToDepthAttrs& spaceToDepthAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    };
    SpaceToDepthExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<CommonSpaceToDepthExecutor>(context);
    };
};

}   // namespace intel_cpu
}   // namespace ov