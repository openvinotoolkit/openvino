// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"
#include "space_to_depth.hpp"

#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_space_to_depth.hpp"
#endif

#include "common/ref_space_to_depth.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct SpaceToDepthExecutorDesc {
    ExecutorType executorType;
    SpaceToDepthExecutorBuilderCPtr builder;
};

const std::vector<SpaceToDepthExecutorDesc>& getSpaceToDepthExecutorsList();

class SpaceToDepthExecutorFactory : public ExecutorFactory {
public:
    SpaceToDepthExecutorFactory(const SpaceToDepthAttrs& paceToDepthParams,
                           const std::vector<MemoryDescPtr>& srcDescs,
                           const std::vector<MemoryDescPtr>& dstDescs,
                           const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getSpaceToDepthExecutorsList()) {
            if (desc.builder->isSupported(paceToDepthParams, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~SpaceToDepthExecutorFactory() = default;
    virtual SpaceToDepthExecutorPtr makeExecutor(const SpaceToDepthAttrs& paceToDepthParams,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs,
                                            const dnnl::primitive_attr &attr) {
        auto build = [&](const SpaceToDepthExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(paceToDepthParams, srcDescs, dstDescs, attr)) {
                return executor;
            }

            SpaceToDepthExecutorPtr ptr = nullptr;
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
    std::vector<SpaceToDepthExecutorDesc> supportedDescs;
    const SpaceToDepthExecutorDesc* chosenDesc = nullptr;
};

using SpaceToDepthExecutorFactoryPtr = std::shared_ptr<SpaceToDepthExecutorFactory>;
using SpaceToDepthExecutorFactoryCPtr = std::shared_ptr<const SpaceToDepthExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov