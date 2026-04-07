// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "pad.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_pad.hpp"
#endif

namespace ov::intel_cpu {

struct PadExecutorDesc {
    ExecutorType executorType;
    PadExecutorBuilderCPtr builder;
};

const std::vector<PadExecutorDesc>& getPadExecutorsList();

class PadExecutorFactory : public ExecutorFactoryLegacy {
public:
    PadExecutorFactory(const PadAttrs& padAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (const auto& desc : getPadExecutorsList()) {
            if (desc.builder->isSupported(padAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~PadExecutorFactory() override = default;

    virtual PadExecutorPtr makeExecutor(const PadAttrs& padAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr& attr) {
        auto build = [&](const PadExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(padAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }

            PadExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported Pad executor is not found");
    }

    bool isEmpty() {
        return supportedDescs.empty();
    }

private:
    std::vector<PadExecutorDesc> supportedDescs;
    const PadExecutorDesc* chosenDesc = nullptr;
};

using PadExecutorFactoryPtr = std::shared_ptr<PadExecutorFactory>;
using PadExecutorFactoryCPtr = std::shared_ptr<const PadExecutorFactory>;

}  // namespace ov::intel_cpu