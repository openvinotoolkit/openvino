// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convert.hpp"
#include "executor.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_convert.hpp"
#endif

#include "common/primitive_cache.hpp"
#include "common/ref_convert.hpp"
#include "onednn/iml_type_mapper.h"

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
        for (auto& desc : getConvertExecutorsList()) {
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
