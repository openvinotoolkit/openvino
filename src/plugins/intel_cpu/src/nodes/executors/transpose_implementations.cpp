// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/common/ref_opt_transpose.hpp"
#include "nodes/executors/common/ref_transpose.hpp"
#include "nodes/executors/common/reorder_transpose.hpp"
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

template <typename ExecutorT>
ExecutorPtr createTransposeExecutor(const TransposeAttrs& attrs,
                                    const MemoryArgs& memory,
                                    const ExecutorContext::CPtr& context) {
    return ExecutorT::create(attrs, memory, context);
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
            "transpose_reorder",
            ExecutorType::Dnnl,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return ReorderTransposeExecutor::supports(config);
            },
            HasNoOptimalConfig<TransposeAttrs>{},
            acceptsAnyShape,
            createTransposeExecutor<ReorderTransposeExecutor>
            )
        OV_CPU_INSTANCE_COMMON(
            "transpose_ref_optimized",
            ExecutorType::Common,
            OperationType::Transpose,
            [](const TransposeConfig& config) -> bool {
                return RefOptimizedTransposeExecutor::supports(config);
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
                return ACLTransposeExecutor::supports(config);
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
                return MlasTransposeExecutor::supports(config);
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
                return JitTransposeExecutor::supports(config);
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
                return RefTransposeExecutor::supports(config);
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
