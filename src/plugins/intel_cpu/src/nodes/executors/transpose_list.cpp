// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_list.hpp"

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/ref_opt_transpose.hpp"
#include "nodes/executors/common/ref_transpose.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "utils/arch_macros.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/jit_transpose.hpp"
#elif defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_ARM64)
#    include "nodes/executors/mlas/mlas_transpose.hpp"
#elif defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_transpose.hpp"
#endif
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

const std::vector<TransposeExecutorDesc>& getTransposeExecutorsList() {
    static const std::vector<TransposeExecutorDesc> descs = {
        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefOptimizedTransposeExecutorBuilder>())
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLTransposeExecutorBuilder>())
                OV_CPU_INSTANCE_MLAS_ARM64(ExecutorType::Mlas, std::make_shared<MlasTransposeExecutorBuilder>())
                    OV_CPU_INSTANCE_X64(ExecutorType::Jit, std::make_shared<JitTransposeExecutorBuilder>())
                        OV_CPU_INSTANCE_COMMON(ExecutorType::Common, std::make_shared<RefTransposeExecutorBuilder>())};

    return descs;
}

TransposeExecutorPtr TransposeExecutorFactory::makeExecutor(const TransposeParams& transposeParams,
                                                            const std::vector<MemoryDescPtr>& srcDescs,
                                                            const std::vector<MemoryDescPtr>& dstDescs,
                                                            const dnnl::primitive_attr& attr) {
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

}  // namespace ov::intel_cpu
