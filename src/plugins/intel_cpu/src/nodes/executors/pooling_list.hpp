// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"
#include "pooling.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_pooling.hpp"
#endif

namespace ov::intel_cpu {

struct PoolingExecutorDesc {
    ExecutorType executorType;
    PoolingExecutorBuilderCPtr builder;
};

const std::vector<PoolingExecutorDesc>& getPoolingExecutorsList();

class PoolingExecutorFactory : public ExecutorFactoryLegacy {
public:
    PoolingExecutorFactory(const PoolingAttrs& poolingAttrs,
                           const std::vector<MemoryDescPtr>& srcDescs,
                           const std::vector<MemoryDescPtr>& dstDescs,
                           const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (auto& desc : getPoolingExecutorsList()) {
            if (desc.builder->isSupported(poolingAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~PoolingExecutorFactory() override = default;
    virtual PoolingExecutorPtr makeExecutor(const PoolingAttrs& poolingAttrs,
                                            const std::vector<MemoryDescPtr>& srcDescs,
                                            const std::vector<MemoryDescPtr>& dstDescs,
                                            const dnnl::primitive_attr& attr) {
        auto build = [&](const PoolingExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(poolingAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }

            PoolingExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported Pooling executor is not found");
    }

private:
    std::vector<PoolingExecutorDesc> supportedDescs;
    const PoolingExecutorDesc* chosenDesc = nullptr;
};

using PoolingExecutorFactoryPtr = std::shared_ptr<PoolingExecutorFactory>;
using PoolingExecutorFactoryCPtr = std::shared_ptr<const PoolingExecutorFactory>;

}  // namespace ov::intel_cpu
