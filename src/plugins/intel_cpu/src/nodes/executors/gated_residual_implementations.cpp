// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "utils/arch_macros.h"

// Only the x64 oneDNN implementation uses these (the OV_CPU_INSTANCE_* macros expand to nothing elsewhere).
#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/debug_messages.hpp"
#    include "nodes/executors/dnnl/dnnl_gated_residual_executor.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/implementation_utils.hpp"
#endif

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<GatedResidualAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<GatedResidualAttrs>> gatedResidualImplementations {
        OV_CPU_INSTANCE_DNNL_X64(
            "gated_residual_dnnl",
            ExecutorType::Dnnl,
            OperationType::GatedResidual,
            [](const GatedResidualConfig& config) -> bool {
                VERIFY(GatedResidualDnnlExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);
                return true;
            },
            HasNoOptimalConfig<GatedResidualAttrs>{},
            AcceptsAnyShape<GatedResidualAttrs>,
            CreateDefault<GatedResidualDnnlExecutor, GatedResidualAttrs>{}
        )
    };

    return gatedResidualImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
