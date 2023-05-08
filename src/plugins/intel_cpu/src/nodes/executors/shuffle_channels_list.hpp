// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"
#include "shuffle_channels.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_shuffle_channels.hpp"
#endif

#include "common/ref_shuffle_channels.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ShuffleChannelsExecutorDesc {
    ExecutorType executorType;
    ShuffleChannelsExecutorBuilderCPtr builder;
};

const std::vector<ShuffleChannelsExecutorDesc>& getShuffleChannelsExecutorsList();

class ShuffleChannelsExecutorFactory : public ExecutorFactory {
public:
    ShuffleChannelsExecutorFactory(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                                const std::vector<MemoryDescPtr>& srcDescs,
                                const std::vector<MemoryDescPtr>& dstDescs,
                                const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getShuffleChannelsExecutorsList()) {
            if (desc.builder->isSupported(shuffleChannelsAttributes, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ShuffleChannelsExecutorFactory() = default;
    virtual ShuffleChannelsExecutorPtr makeExecutor(const ShuffleChannelsAttributes& shuffleChannelsAttributes,
                                                 const std::vector<MemoryDescPtr>& srcDescs,
                                                 const std::vector<MemoryDescPtr>& dstDescs,
                                                 const dnnl::primitive_attr &attr) {
        auto build = [&](const ShuffleChannelsExecutorDesc* desc) {
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(shuffleChannelsAttributes, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ShuffleChannelsExecutorPtr ptr = nullptr;
            return ptr;
        };


        if (chosenDesc) {
            if (auto executor = build(chosenDesc)) {
                return executor;
            }
        }

        for (const auto& sd : supportedDescs) {
            if (auto executor = build(&sd)) {
                chosenDesc = &sd;
                return executor;
            }
        }

        IE_THROW() << "Supported executor is not found";
    }

private:
    std::vector<ShuffleChannelsExecutorDesc> supportedDescs;
    const ShuffleChannelsExecutorDesc* chosenDesc = nullptr;
};

using ShuffleChannelsExecutorFactoryPtr = std::shared_ptr<ShuffleChannelsExecutorFactory>;
using ShuffleChannelsExecutorFactoryCPtr = std::shared_ptr<const ShuffleChannelsExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov