// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/common/paged_selective_ssm_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "utils/arch_macros.h"

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<PagedSelectiveSSMAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<PagedSelectiveSSMAttrs>> implementations {
        OV_CPU_INSTANCE_COMMON(
            "paged_selective_ssm_common_executor",
            ExecutorType::Common,
            OperationType::PagedSelectiveSSM,
            [](const PagedSelectiveSSMConfig& config) -> bool {
                return PagedSelectiveSSMExecutor::supports(config);
            },
            HasNoOptimalConfig<PagedSelectiveSSMAttrs>{},
            AcceptsAnyShape<PagedSelectiveSSMAttrs>,
            CreateDefault<PagedSelectiveSSMExecutor, PagedSelectiveSSMAttrs>{}
        )
    };

    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
