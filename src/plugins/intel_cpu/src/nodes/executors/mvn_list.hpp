// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "mvn.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_mvn.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct MVNExecutorDesc {
    ExecutorType executorType;
    MVNExecutorBuilderCPtr builder;
};

const std::vector<MVNExecutorDesc>& getMVNExecutorsList();

class MVNExecutorFactory : public ExecutorFactory {
public:
    MVNExecutorFactory(const MVNAttrs& mvnAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getMVNExecutorsList()) {
            if (desc.builder->isSupported(mvnAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~MVNExecutorFactory() = default;
    virtual MVNExecutorPtr makeExecutor(const MVNAttrs& mvnAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const MVNExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(mvnAttrs, srcDescs, dstDescs, attr)) {
                return executor;
            }

            MVNExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported MVN executor is not found");
    }

    bool isEmpty() {
        return supportedDescs.empty();
    }

private:
    std::vector<MVNExecutorDesc> supportedDescs;
    const MVNExecutorDesc* chosenDesc = nullptr;
};

using MVNExecutorFactoryPtr = std::shared_ptr<MVNExecutorFactory>;
using MVNExecutorFactoryCPtr = std::shared_ptr<const MVNExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov