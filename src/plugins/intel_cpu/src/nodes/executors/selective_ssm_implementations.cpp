// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/common/selective_ssm_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "utils/arch_macros.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/selective_ssm_jit_executor.hpp"
#endif

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<SelectiveSSMAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<SelectiveSSMAttrs>> implementations {
        OV_CPU_INSTANCE_X64(
            "selective_ssm_jit_executor",
            ExecutorType::Jit,
            OperationType::SelectiveSSM,
            [](const SelectiveSSMConfig& config) -> bool {
                return SelectiveSSMJitExecutor::supports(config);
            },
            HasNoOptimalConfig<SelectiveSSMAttrs>{},
            AcceptsAnyShape<SelectiveSSMAttrs>,
            CreateDefault<SelectiveSSMJitExecutor, SelectiveSSMAttrs>{}
        )
        OV_CPU_INSTANCE_COMMON(
            "selective_ssm_common_executor",
            ExecutorType::Common,
            OperationType::SelectiveSSM,
            [](const SelectiveSSMConfig& config) -> bool {
                return SelectiveSSMExecutor::supports(config);
            },
            HasNoOptimalConfig<SelectiveSSMAttrs>{},
            AcceptsAnyShape<SelectiveSSMAttrs>,
            CreateDefault<SelectiveSSMExecutor, SelectiveSSMAttrs>{}
        )
    };

    return implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
