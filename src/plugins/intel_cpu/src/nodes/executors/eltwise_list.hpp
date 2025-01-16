// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor.hpp"

#include "eltwise.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "aarch64/jit_eltwise.hpp"
#include "acl/acl_eltwise.hpp"
#endif
#if defined(OV_CPU_WITH_SHL)
#include "shl/shl_eltwise.hpp"
#endif

#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

struct EltwiseExecutorDesc {
    ExecutorType executorType;
    EltwiseExecutorBuilderCPtr builder;
};

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList();

class EltwiseExecutorFactory : public ExecutorFactoryLegacy {
public:
    EltwiseExecutorFactory(const EltwiseAttrs& eltwiseAttrs,
                       const std::vector<MemoryDescPtr>& srcDescs,
                       const std::vector<MemoryDescPtr>& dstDescs,
                       const ExecutorContext::CPtr context) : ExecutorFactoryLegacy(context) {
        for (auto& desc : getEltwiseExecutorsList()) {
            if (desc.builder->isSupported(eltwiseAttrs, srcDescs, dstDescs)) {
                supportedDescs.push_back(desc);
            }
        }
    }

    ~EltwiseExecutorFactory() = default;
    virtual EltwiseExecutorPtr makeExecutor(const EltwiseAttrs& eltwiseAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs,
                                        const std::vector<EltwisePostOp>& postOps) {
        auto build = [&](const EltwiseExecutorDesc* desc) {
            auto executor = desc->builder->makeExecutor(context);
            if (executor->init(eltwiseAttrs, srcDescs, dstDescs, postOps)) {
                return executor;
            }

            EltwiseExecutorPtr ptr = nullptr;
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

        OPENVINO_THROW("Supported Eltwise executor is not found");
    }

    bool isEmpty() {
        return supportedDescs.empty();
    }

private:
    std::vector<EltwiseExecutorDesc> supportedDescs;
    const EltwiseExecutorDesc* chosenDesc = nullptr;
};

using EltwiseExecutorFactoryPtr = std::shared_ptr<EltwiseExecutorFactory>;
using EltwiseExecutorFactoryCPtr = std::shared_ptr<const EltwiseExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov
