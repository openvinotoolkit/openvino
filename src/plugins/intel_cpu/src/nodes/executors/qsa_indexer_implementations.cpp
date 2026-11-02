// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/common/qsa_indexer_ref_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "utils/arch_macros.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/qsa_indexer_jit_executor.hpp"
#endif

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<QSAIndexerAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<QSAIndexerAttrs>> implementations {
        OV_CPU_INSTANCE_X64(
            "qsa_indexer_jit_executor",
            ExecutorType::Jit,
            OperationType::QSAIndexer,
            [](const QSAIndexerConfig& config) -> bool {
                return QSAIndexerJitExecutor::supports(config);
            },
            HasNoOptimalConfig<QSAIndexerAttrs>{},
            AcceptsAnyShape<QSAIndexerAttrs>,
            CreateDefault<QSAIndexerJitExecutor, QSAIndexerAttrs>{}
        )
        OV_CPU_INSTANCE_COMMON(
            "qsa_indexer_ref_executor",
            ExecutorType::Reference,
            OperationType::QSAIndexer,
            [](const QSAIndexerConfig& config) -> bool {
                return QSAIndexerRefExecutor::supports(config);
            },
            HasNoOptimalConfig<QSAIndexerAttrs>{},
            AcceptsAnyShape<QSAIndexerAttrs>,
            CreateDefault<QSAIndexerRefExecutor, QSAIndexerAttrs>{}
        )
    };

    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
