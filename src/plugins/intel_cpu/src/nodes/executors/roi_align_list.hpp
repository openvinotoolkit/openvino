// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "roi_align.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_roi_align.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct ROIAlignExecutorDesc {
    ExecutorType executorType;
    ROIAlignExecutorBuilderCPtr builder;
};

const std::vector<ROIAlignExecutorDesc>& getROIAlignExecutorsList();

class ROIAlignExecutorFactory : public ExecutorFactory {
public:
    ROIAlignExecutorFactory(const ROIAlignAttrs& roialignAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getROIAlignExecutorsList()) {
            if (desc.builder->isSupported(roialignAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~ROIAlignExecutorFactory() = default;
    virtual ROIAlignExecutorPtr makeExecutor(const ROIAlignAttrs& roialignAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const ROIAlignExecutorDesc* desc) {
            switch (desc->executorType) {
                default: {
                    auto executor = desc->builder->makeExecutor(context);
                    if (executor->init(roialignAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
            }

            ROIAlignExecutorPtr ptr = nullptr;
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
    std::vector<ROIAlignExecutorDesc> supportedDescs;
    const ROIAlignExecutorDesc* chosenDesc = nullptr;
};

using ROIAlignExecutorFactoryPtr = std::shared_ptr<ROIAlignExecutorFactory>;
using ROIAlignExecutorFactoryCPtr = std::shared_ptr<const ROIAlignExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov