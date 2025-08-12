// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "reduce.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_reduce.hpp"
#endif

namespace ov::intel_cpu {

struct ReduceExecutorDesc {
    ExecutorType executorType;
    ReduceExecutorBuilderCPtr builder;
};

const std::vector<ReduceExecutorDesc>& getReduceExecutorsList();

class ReduceExecutorFactory : public ExecutorFactoryLegacy {
public:
    ReduceExecutorFactory(const ReduceAttrs& reduceAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (const auto& desc : getReduceExecutorsList()) {
            if (desc.builder->isSupported(reduceAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ReduceExecutorFactory() override = default;
    virtual ReduceExecutorPtr makeExecutor(const ReduceAttrs& reduceAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs,
                                           const dnnl::primitive_attr& attr) {
        auto build = [&](const ReduceExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(reduceAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }

            ReduceExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported Reduce executor is not found");
    }

    bool isEmpty() {
        return supportedDescs.empty();
    }

private:
    std::vector<ReduceExecutorDesc> supportedDescs;
    const ReduceExecutorDesc* chosenDesc = nullptr;
};

using ReduceExecutorFactoryPtr = std::shared_ptr<ReduceExecutorFactory>;
using ReduceExecutorFactoryCPtr = std::shared_ptr<const ReduceExecutorFactory>;

}  // namespace ov::intel_cpu
