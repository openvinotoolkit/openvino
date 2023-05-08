// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/shuffle_channels.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLShuffleChannelsExecutor : public ShuffleChannelsExecutor {
public:
    explicit ACLShuffleChannelsExecutor(const ExecutorContext::CPtr context);
    bool init(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implDescType; }
    ~ACLShuffleChannelsExecutor() override = default;
private:
    ShuffleChannelsAttributes aclShuffleChannelsAttributes;
    impl_desc_type implDescType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NEActivationLayer> aclActivationLayer = nullptr;
    std::unique_ptr<arm_compute::NEChannelShuffleLayer> aclChannelShuffleLayer = nullptr;
};

class ACLShuffleChannelsExecutorBuilder : public ShuffleChannelsExecutorBuilder {
public:
    ~ACLShuffleChannelsExecutorBuilder() = default;
    bool isSupported(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    };
    ShuffleChannelsExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLShuffleChannelsExecutor>(context);
    };
};

}   // namespace intel_cpu
}   // namespace ov