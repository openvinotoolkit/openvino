// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "deconv.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_deconv.hpp"
#endif

#include "dnnl/dnnl_deconv.hpp"

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct DeconvExecutorDesc {
    ExecutorType executorType;
    DeconvExecutorBuilderCPtr builder;
};

const std::vector<DeconvExecutorDesc>& getDeconvExecutorsList();

class DeconvExecutorFactory : public ExecutorFactory {
public:
    DeconvExecutorFactory(const DeconvAttrs& deconvAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactory(context) {
        for (auto& desc : getDeconvExecutorsList()) {
            if (desc.builder->isSupported(deconvAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~DeconvExecutorFactory() = default;
    virtual DeconvExecutorPtr makeExecutor(const DeconvAttrs& deconvAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const dnnl::primitive_attr &attr) {
        auto build = [&](const DeconvExecutorDesc* desc) {
            switch (desc->executorType) {
#if defined(OPENVINO_ARCH_X86_64)
                case ExecutorType::x64: {
                    auto builder = [&](const DnnlDeconvExecutor::Key& key) -> DeconvExecutorPtr {
                        auto executor = desc->builder->makeExecutor();
                        if (executor->init(deconvAttrs, srcDescs, dstDescs, attr)) {
                            return executor;
                        } else {
                            return nullptr;
                        }
                    };

                    auto key = DnnlDeconvExecutor::Key(deconvAttrs, srcDescs, dstDescs, attr);
                    auto res = runtimeCache->getOrCreate(key, builder);
                    return res.first;
                } break;
#endif
                default: {
                    auto executor = desc->builder->makeExecutor();
                    if (executor->init(deconvAttrs, srcDescs, dstDescs, attr)) {
                        return executor;
                    }
                } break;
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

        IE_THROW() << "Supported executor is not found";
    }

private:
    std::vector<DeconvExecutorDesc> supportedDescs;
    const DeconvExecutorDesc* chosenDesc = nullptr;
};

using DeconvExecutorFactoryPtr = std::shared_ptr<DeconvExecutorFactory>;
using DeconvExecutorFactoryCPtr = std::shared_ptr<const DeconvExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov