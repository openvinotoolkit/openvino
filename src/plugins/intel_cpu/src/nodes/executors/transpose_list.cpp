// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<TransposeExecutorDesc>& getTransposeExecutorsList() {
    static const std::vector<TransposeExecutorDesc> descs = {
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefOptimizedTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_MLAS_ARM64(ExecutorType::Mlas, std::make_shared<MlasTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_X64(ExecutorType::jit_x64, std::make_shared<JitTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefTransposeExecutorBuilder>())
    };

    return descs;
}

TransposeExecutorPtr TransposeExecutorFactory::makeExecutor(const TransposeParams& transposeParams,
                                                           const std::vector<MemoryDescPtr>& srcDescs,
                                                           const std::vector<MemoryDescPtr>& dstDescs,
                                                           const dnnl::primitive_attr &attr) {
    auto build = [&](const TransposeExecutorDesc* desc) {
        auto executor = desc->builder->makeExecutor(context);
        if (executor->init(transposeParams, srcDescs, dstDescs, attr)) {
            return executor;
        }
        TransposeExecutorPtr ptr = nullptr;
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

}   // namespace intel_cpu
}   // namespace ov
