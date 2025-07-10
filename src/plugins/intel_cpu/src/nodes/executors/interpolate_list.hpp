// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "interpolate.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_interpolate.hpp"
#endif

namespace ov::intel_cpu {

struct InterpolateExecutorDesc {
    ExecutorType executorType;
    InterpolateExecutorBuilderCPtr builder;
};

const std::vector<InterpolateExecutorDesc>& getInterpolateExecutorsList();

class InterpolateExecutorFactory : public ExecutorFactoryLegacy {
public:
    InterpolateExecutorFactory(const InterpolateAttrs& InterpolateAttrs,
                               const std::vector<MemoryDescPtr>& srcDescs,
                               const std::vector<MemoryDescPtr>& dstDescs,
                               const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (const auto& desc : getInterpolateExecutorsList()) {
            if (desc.builder->isSupported(InterpolateAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~InterpolateExecutorFactory() override = default;
    virtual InterpolateExecutorPtr makeExecutor(const InterpolateAttrs& interpolateAttrs,
                                                const std::vector<MemoryDescPtr>& srcDescs,
                                                const std::vector<MemoryDescPtr>& dstDescs,
                                                const dnnl::primitive_attr& attr) {
        auto build = [&](const InterpolateExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(interpolateAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }

            InterpolateExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported Interpolate executor is not found");
    }

    bool isEmpty() {
        return supportedDescs.empty();
    }

private:
    std::vector<InterpolateExecutorDesc> supportedDescs;
    const InterpolateExecutorDesc* chosenDesc = nullptr;
};

using InterpolateExecutorFactoryPtr = std::shared_ptr<InterpolateExecutorFactory>;
using InterpolateExecutorFactoryCPtr = std::shared_ptr<const InterpolateExecutorFactory>;

}  // namespace ov::intel_cpu
