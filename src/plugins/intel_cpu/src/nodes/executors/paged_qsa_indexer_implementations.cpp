// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/common/paged_qsa_indexer_ref_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/paged_qsa_indexer_config.hpp"
#include "utils/arch_macros.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/paged_qsa_indexer_jit_executor.hpp"
#endif

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<PagedQSAIndexerAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<PagedQSAIndexerAttrs>> implementations {
        OV_CPU_INSTANCE_X64(
            "paged_qsa_indexer_jit_executor",
            ExecutorType::Jit,
            OperationType::PagedQSAIndexer,
            [](const PagedQSAIndexerConfig& config) -> bool {
                return PagedQSAIndexerJitExecutor::supports(config);
            },
            HasNoOptimalConfig<PagedQSAIndexerAttrs>{},
            AcceptsAnyShape<PagedQSAIndexerAttrs>,
            CreateDefault<PagedQSAIndexerJitExecutor, PagedQSAIndexerAttrs>{}
        )
        OV_CPU_INSTANCE_COMMON(
            "paged_qsa_indexer_ref_executor",
            ExecutorType::Reference,
            OperationType::PagedQSAIndexer,
            [](const PagedQSAIndexerConfig& config) -> bool {
                return PagedQSAIndexerRefExecutor::supports(config);
            },
            HasNoOptimalConfig<PagedQSAIndexerAttrs>{},
            AcceptsAnyShape<PagedQSAIndexerAttrs>,
            CreateDefault<PagedQSAIndexerRefExecutor, PagedQSAIndexerAttrs>{}
        )
    };

    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
