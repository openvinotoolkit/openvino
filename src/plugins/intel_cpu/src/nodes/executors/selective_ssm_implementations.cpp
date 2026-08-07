// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/ref/selective_ssm_ref_executor.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "utils/arch_macros.h"

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<SelectiveSSMAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<SelectiveSSMAttrs>> selectiveSSMImplementations {
        OV_CPU_INSTANCE_COMMON(
            "selective_ssm_ref_executor",
            ExecutorType::Common,
            OperationType::SelectiveSSM,
            [](const SelectiveSSMConfig& config) -> bool {
                return SelectiveSSMRefExecutor::supports(config);
            },
            HasNoOptimalConfig<SelectiveSSMAttrs>{},
            AcceptsAnyShape<SelectiveSSMAttrs>,
            CreateDefault<SelectiveSSMRefExecutor, SelectiveSSMAttrs>{}
        )
    };

    return selectiveSSMImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
