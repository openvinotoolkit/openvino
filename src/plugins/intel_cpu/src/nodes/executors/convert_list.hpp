// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "convert.hpp"
#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_convert.hpp"
#endif

namespace ov::intel_cpu {

struct ConvertExecutorDesc {
    ExecutorType executorType;
    ConvertExecutorBuilderCPtr builder;
};

const std::vector<ConvertExecutorDesc>& getConvertExecutorsList();

class ConvertExecutorFactory : public ExecutorFactoryLegacy {
public:
    ConvertExecutorFactory(const ConvertParams& convertParams,
                           const MemoryDescPtr& srcDesc,
                           const MemoryDescPtr& dstDesc,
                           const ExecutorContext::CPtr& context)
        : ExecutorFactoryLegacy(context) {
        for (const auto& desc : getConvertExecutorsList()) {
            if (desc.builder->isSupported(convertParams, srcDesc, dstDesc)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ConvertExecutorFactory() override = default;
    virtual ConvertExecutorPtr makeExecutor(const ConvertParams& convertParams,
                                            const MemoryDescPtr& srcDesc,
                                            const MemoryDescPtr& dstDesc,
                                            const dnnl::primitive_attr& attr);

private:
    std::vector<ConvertExecutorDesc> supportedDescs;
    const ConvertExecutorDesc* chosenDesc = nullptr;
};

using ConvertExecutorFactoryPtr = std::shared_ptr<ConvertExecutorFactory>;
using ConvertExecutorFactoryCPtr = std::shared_ptr<const ConvertExecutorFactory>;

}  // namespace ov::intel_cpu
