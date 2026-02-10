// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_list.hpp"

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "openvino/core/except.hpp"
#include "split.hpp"
#if defined(OV_CPU_WITH_ACL)
#    include "acl/acl_split.hpp"
#endif
#include "utils/arch_macros.h"

namespace ov::intel_cpu {

const std::vector<SplitExecutorDesc>& getSplitExecutorsList() {
    static const std::vector<SplitExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclSplitExecutorBuilder>())};
    return descs;
}

SplitExecutorPtr SplitExecutorFactory::makeExecutor(const SplitAttrs& splitAttrs,
                                                    const std::vector<MemoryDescPtr>& srcDescs,
                                                    const std::vector<MemoryDescPtr>& dstDescs,
                                                    const dnnl::primitive_attr& attr) {
    auto build = [&](const SplitExecutorDesc* desc) {
        auto executor = desc->builder->makeExecutor(context);
        if (executor->init(splitAttrs, srcDescs, dstDescs, attr)) {
            return executor;
        }
        return SplitExecutorPtr{};
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

    OPENVINO_THROW("Supported Split executor is not found");
}

}  // namespace ov::intel_cpu
