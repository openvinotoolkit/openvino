// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "deconv.hpp"
#include "executor.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_deconv.hpp"
#endif

#include "common/primitive_cache.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

struct DeconvExecutorDesc {
    ExecutorType executorType;
    DeconvExecutorBuilderCPtr builder;
};

const std::vector<DeconvExecutorDesc>& getDeconvExecutorsList();

class DeconvExecutorFactory : public ExecutorFactoryLegacy {
public:
    DeconvExecutorFactory(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (auto& desc : getDeconvExecutorsList()) {
            if (desc.builder->isSupported(deconvAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~DeconvExecutorFactory() override = default;
    virtual DeconvExecutorPtr makeExecutor(const DeconvAttrs& deconvAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs,
                                           const dnnl::primitive_attr& attr) {
        auto build = [&](const DeconvExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(deconvAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }
            DeconvExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("DeconvExecutorFactory: Supported executor is not found");
    }

private:
    std::vector<DeconvExecutorDesc> supportedDescs;
    const DeconvExecutorDesc* chosenDesc = nullptr;
};

using DeconvExecutorFactoryPtr = std::shared_ptr<DeconvExecutorFactory>;
using DeconvExecutorFactoryCPtr = std::shared_ptr<const DeconvExecutorFactory>;

}  // namespace ov::intel_cpu
