// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_list.hpp"

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/ref_convert.hpp"
#include "nodes/executors/convert.hpp"
#include "nodes/executors/executor.hpp"
#include "openvino/core/except.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_convert.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<ConvertExecutorDesc>& getConvertExecutorsList() {
    static std::vector<ConvertExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLConvertExecutorBuilder>())
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<CommonConvertExecutorBuilder>())};

    return descs;
}

ConvertExecutorPtr ConvertExecutorFactory::makeExecutor(const ConvertParams& convertParams,
                                                        const MemoryDescPtr& srcDesc,
                                                        const MemoryDescPtr& dstDesc,
                                                        const dnnl::primitive_attr& attr) {
    auto build = [&](const ConvertExecutorDesc* desc) {
        auto executor = desc->builder->makeExecutor(context);
        if (executor->init(convertParams, srcDesc, dstDesc, attr)) {
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

    OPENVINO_THROW("Supported executor is not found");
}

}  // namespace ov::intel_cpu
