// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/shuffle_channels.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov {
namespace intel_cpu {

class CommonShuffleChannelsExecutor : public ShuffleChannelsExecutor {
public:
    explicit CommonShuffleChannelsExecutor(const ExecutorContext::CPtr context);
    bool init(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return commonShuffleChannelsAttributes.implDescType; }
    ~CommonShuffleChannelsExecutor() override = default;
private:
    ShuffleChannelsAttributes commonShuffleChannelsAttributes;
    std::unique_ptr<PermuteKernel> permuteKernel = nullptr;
};

class CommonShuffleChannelsExecutorBuilder : public ShuffleChannelsExecutorBuilder {
public:
    ~CommonShuffleChannelsExecutorBuilder() = default;
    bool isSupported(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    };
    ShuffleChannelsExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<CommonShuffleChannelsExecutor>(context);
    };
};

}   // namespace intel_cpu
}   // namespace ov