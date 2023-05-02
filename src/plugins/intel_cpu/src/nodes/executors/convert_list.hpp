// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "convert.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_convert.hpp"
#endif

#include "common/ref_convert.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ConvertExecutorDesc {
    ExecutorType executorType;
    ConvertExecutorBuilderCPtr builder;
};

const std::vector<ConvertExecutorDesc>& getConvertExecutorsList();

class ConvertExecutorFactory : public ExecutorFactory {
public:
    ConvertExecutorFactory(const ConvertParams& convertParams,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getConvertExecutorsList()) {
            if (desc.builder->isSupported(convertParams, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ConvertExecutorFactory() = default;
    virtual ConvertExecutorPtr makeExecutor(const ConvertParams& convertParams,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs,
                                              const dnnl::primitive_attr &attr) {
        auto build = [&](const ConvertExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(convertParams, srcDescs, dstDescs, attr)) {
                return executor;
            }
            ConvertExecutorPtr ptr = nullptr;
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
    std::vector<ConvertExecutorDesc> supportedDescs;
    const ConvertExecutorDesc* chosenDesc = nullptr;
};

using ConvertExecutorFactoryPtr = std::shared_ptr<ConvertExecutorFactory>;
using ConvertExecutorFactoryCPtr = std::shared_ptr<const ConvertExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov