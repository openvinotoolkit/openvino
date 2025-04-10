// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_list.hpp"

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
