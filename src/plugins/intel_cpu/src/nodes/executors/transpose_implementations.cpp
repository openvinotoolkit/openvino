// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/ref_opt_transpose.hpp"
#include "nodes/executors/common/ref_transpose.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/transpose_config.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_transpose.hpp"
#endif

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_ARM64)
#    include "nodes/executors/mlas/mlas_transpose.hpp"
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/jit_transpose.hpp"
#endif

namespace ov::intel_cpu {

namespace {

std::vector<MemoryDescPtr> srcDescs(const MemoryDescArgs& descs) {
    return {descs.at(ARG_SRC)};
}

std::vector<MemoryDescPtr> dstDescs(const MemoryDescArgs& descs) {
    return {descs.at(ARG_DST)};
}

template <typename Builder>
bool isSupportedByBuilder(const TransposeConfig& config) {
    Builder builder;
    return builder.isSupported(config.attrs.params, srcDescs(config.descs), dstDescs(config.descs));
}

template <typename ExecutorT>
ExecutorPtr createTransposeExecutor(const TransposeAttrs& attrs,
                                    const MemoryArgs& memory,
                                    const ExecutorContext::CPtr& context) {
    const auto& descs = attrs.descs.empty() ? memoryDescsFromMemory(memory) : attrs.descs;

    auto executor = std::make_shared<ExecutorT>(context);
    dnnl::primitive_attr attr;
    if (!executor->init(attrs.params, {descs.at(ARG_SRC)}, {descs.at(ARG_DST)}, attr)) {
        return nullptr;
    }
    return executor;
}

const auto acceptsAnyShape = []([[maybe_unused]] const TransposeAttrs& attrs, [[maybe_unused]] const MemoryArgs& mem) {
    return true;
};

}  // namespace

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<TransposeAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<TransposeAttrs>> transposeImplementations {
        OV_CPU_INSTANCE_COMMON(
            "transpose_ref_optimized",
            ExecutorType::Common,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return isSupportedByBuilder<RefOptimizedTransposeExecutorBuilder>(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<RefOptimizedTransposeExecutor>
            )
        OV_CPU_INSTANCE_ACL(
            "transpose_acl",
            ExecutorType::Acl,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return isSupportedByBuilder<ACLTransposeExecutorBuilder>(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<ACLTransposeExecutor>
            )
        OV_CPU_INSTANCE_MLAS_ARM64(
            "transpose_mlas",
            ExecutorType::Mlas,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return isSupportedByBuilder<MlasTransposeExecutorBuilder>(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<MlasTransposeExecutor>
            )
        OV_CPU_INSTANCE_X64(
            "transpose_jit",
            ExecutorType::Jit,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return isSupportedByBuilder<JitTransposeExecutorBuilder>(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<JitTransposeExecutor>
            )
        OV_CPU_INSTANCE_COMMON(
            "transpose_ref",
            ExecutorType::Common,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return isSupportedByBuilder<RefTransposeExecutorBuilder>(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<RefTransposeExecutor>
            )
    };

    return transposeImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
