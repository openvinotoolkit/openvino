// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/gated_delta_net_config.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/ref/gdn_ref_executor.hpp"
#include "utils/arch_macros.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include <cstddef>

#    include "nodes/executors/debug_messages.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "nodes/executors/x64/gdn_jit_executor.hpp"
#    include "openvino/core/type/element_type.hpp"
#endif

namespace ov::intel_cpu {

// clang-format off
template <>
const std::vector<ExecutorImplementation<GatedDeltaNetAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<GatedDeltaNetAttrs>> gatedDeltaNetImplementations {
        OV_CPU_INSTANCE_X64(
            "gdn_jit_executor",
            ExecutorType::Jit,
            OperationType::GatedDeltaNet,
            [](const GatedDeltaNetConfig& config) -> bool {
                return GdnJitExecutor::supports(config);
            },
            HasNoOptimalConfig<GatedDeltaNetAttrs>{},
            [](const GatedDeltaNetAttrs& attrs, const MemoryArgs& memory) -> bool {
                const auto precision = memory.at(ARG_GDN_QUERY)->getDescPtr()->getPrecision();
                const auto& queryShape = memory.at(ARG_GDN_QUERY)->getDescPtr()->getShape();
                const size_t jitVTile = precision == ov::element::f32 ? 1 : attrs.jit_v_tile;
                const auto& valueShape = memory.at(ARG_GDN_VALUE)->getDescPtr()->getShape();
                VERIFY(!queryShape.isDynamic() && !valueShape.isDynamic(), UNSUPPORTED_BY_EXECUTOR, " dynamic shape");

                const auto& queryDims = queryShape.getStaticDims();
                VERIFY(!queryDims.empty(), UNSUPPORTED_BY_EXECUTOR, " empty query dims");

                VERIFY((precision != ov::element::f16 && precision != ov::element::bf16) || queryDims.back() % 32 == 0,
                       UNSUPPORTED_BY_EXECUTOR,
                       " xf16 requires qk_head_size % 32 == 0");

                const auto& valueDims = valueShape.getStaticDims();
                VERIFY(!valueDims.empty(), UNSUPPORTED_BY_EXECUTOR, " empty value dims");
                VERIFY(valueDims.back() % jitVTile == 0,
                       UNSUPPORTED_BY_EXECUTOR,
                       " value head dim is not divisible by JIT tile");
                return true;
            },
            CreateDefault<GdnJitExecutor, GatedDeltaNetAttrs>{}
        )
        OV_CPU_INSTANCE_COMMON(
            "gdn_ref_executor",
            ExecutorType::Common,
            OperationType::GatedDeltaNet,
            [](const GatedDeltaNetConfig& config) -> bool {
                return GdnRefExecutor::supports(config);
            },
            HasNoOptimalConfig<GatedDeltaNetAttrs>{},
            AcceptsAnyShape<GatedDeltaNetAttrs>,
            CreateDefault<GdnRefExecutor, GatedDeltaNetAttrs>{}
        )
    };

    return gatedDeltaNetImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
