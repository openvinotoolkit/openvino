// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "normalize.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_normalize.hpp"
#endif
#include "x64/jit_normalize.hpp"
#include "common/ref_normalize.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct NormalizeL2ExecutorDesc {
    ExecutorType executorType;
    NormalizeL2ExecutorBuilderCPtr builder;
};

const std::vector<NormalizeL2ExecutorDesc>& getNormalizeL2ExecutorsList();

class NormalizeL2ExecutorFactory : public ExecutorFactory {
public:
    NormalizeL2ExecutorFactory(const NormalizeL2Attrs& normalizeL2Attrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getNormalizeL2ExecutorsList()) {
            if (desc.builder->isSupported(normalizeL2Attrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~NormalizeL2ExecutorFactory() = default;
    virtual NormalizeL2ExecutorPtr makeExecutor(const NormalizeL2Attrs& normalizeL2Attrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const NormalizeL2ExecutorDesc* desc) {
            //TODO: enable exeuctor cache for JIT executor
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(normalizeL2Attrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            NormalizeL2ExecutorPtr ptr = nullptr;
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
    std::vector<NormalizeL2ExecutorDesc> supportedDescs;
    const NormalizeL2ExecutorDesc* chosenDesc = nullptr;
};

using NormalizeL2ExecutorFactoryPtr = std::shared_ptr<NormalizeL2ExecutorFactory>;
using NormalizeL2ExecutorFactoryCPtr = std::shared_ptr<const NormalizeL2ExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov