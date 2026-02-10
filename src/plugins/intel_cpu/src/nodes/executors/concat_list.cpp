// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concat_list.hpp"

#if defined(OV_CPU_WITH_ACL)
#    include <memory>
#endif
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/concat.hpp"
#include "nodes/executors/executor.hpp"
#include "openvino/core/except.hpp"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_concat.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<ConcatExecutorDesc>& getConcatExecutorsList() {
    static const std::vector<ConcatExecutorDesc> descs = {
#if defined(OV_CPU_WITH_ACL)
        {ExecutorType::Acl, std::make_shared<AclConcatExecutorBuilder>()},
#endif
    };
    return descs;
}

ConcatExecutorFactory::ConcatExecutorFactory(const ConcatAttrs& concatAttrs,
                                             const std::vector<MemoryDescPtr>& srcDescs,
                                             const std::vector<MemoryDescPtr>& dstDescs,
                                             const ExecutorContext::CPtr& context)
    : ExecutorFactoryLegacy(context) {
    for (const auto& desc : getConcatExecutorsList()) {
        if (desc.builder->isSupported(concatAttrs, srcDescs, dstDescs)) {
            supportedDescs.push_back(desc);
        }
    }
}

ConcatExecutorPtr ConcatExecutorFactory::makeExecutor(const ConcatAttrs& concatAttrs,
                                                      const std::vector<MemoryDescPtr>& srcDescs,
                                                      const std::vector<MemoryDescPtr>& dstDescs,
                                                      const dnnl::primitive_attr& attr) {
    auto build = [&](const ConcatExecutorDesc* desc) -> ConcatExecutorPtr {
        auto executor = desc->builder->makeExecutor(context);
        if (executor->init(concatAttrs, srcDescs, dstDescs, attr)) {
            return executor;
        }
        return nullptr;
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

    OPENVINO_THROW("Supported Concat executor is not found");
}

}  // namespace ov::intel_cpu
