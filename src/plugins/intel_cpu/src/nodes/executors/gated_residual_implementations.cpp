// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <vector>

#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/gated_residual_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "utils/arch_macros.h"

// Only the x64 oneDNN implementation uses these (the OV_CPU_INSTANCE_* macros expand to nothing elsewhere).
#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
#    include "memory_desc/cpu_memory_desc.h"
#    include "nodes/executors/debug_messages.hpp"
#    include "nodes/executors/dnnl/dnnl_gated_residual_executor.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/implementation_utils.hpp"
#    include "nodes/executors/precision_translation.hpp"
#    include "nodes/executors/type_mask.hpp"
#    include "openvino/core/type/element_type.hpp"
#    include "openvino/runtime/system_conf.hpp"
#endif

namespace ov::intel_cpu {

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
using namespace ov::element;
using namespace TypeMaskAlias;

// clang-format off
// {residual, gain, w_down, w_up, src_4, mixed, write_gate}. src_4 is w_inject with float weights and scale_down with
// compressed ones, which the rows tell apart by the w_down type. The other scale, zero-point and compressed-inject ports
// are not mapped and keep their types.
static const TypeMapping dnnlGatedResidualTypeMapping {
    // compressed weights and their scales keep their types
    {{_f32, _any, _u8 | _i8 | _u4 | _i4, _any, _any, _any, _any},
     {bypass(), use<0>(), bypass(), bypass(), bypass(), use<0>(), use<0>()}},
    // bf16 with compressed weights needs native bf16, as in the FullyConnected mapping: without it oneDNN cannot
    // build the bf16-output projection from f32 activations and int weights
    {{_bf16, _any, _u8 | _i8 | _u4 | _i4, _any, _any, _any, _any},
     {bypass(), use<0>(), bypass(), bypass(), bypass(), use<0>(), use<0>()},
     []() { return ov::with_cpu_x86_bfloat16(); }},
    {{_bf16, _any, _u8 | _i8 | _u4 | _i4, _any, _any, _any, _any},
     {just<f32>(), just<f32>(), bypass(), bypass(), bypass(), just<f32>(), just<f32>()}},
    // float weights take the data precision
    {{_f32 | _bf16, _any, _any, _any, _any, _any, _any},
     {bypass(), use<0>(), use<0>(), use<0>(), use<0>(), use<0>(), use<0>()}},
    // f16 runs in f32: the FC projections do not support it
    {{_any, _any, _u8 | _i8 | _u4 | _i4, _any, _any, _any, _any},
     {just<f32>(), just<f32>(), bypass(), bypass(), bypass(), just<f32>(), just<f32>()}},
    {{_any, _any, _any, _any, _any, _any, _any},
     {just<f32>(), just<f32>(), just<f32>(), just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
};

static const MappingNotation gatedResidualMappingNotation {
    {gatedResidualSrcArg(0), 0},
    {gatedResidualSrcArg(1), 1},
    {gatedResidualSrcArg(2), 2},
    {gatedResidualSrcArg(3), 3},
    {gatedResidualSrcArg(4), 4},
    {gatedResidualDstArg(0), 5},
    {gatedResidualDstArg(1), 6},
};
// clang-format on
#endif

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
            [](const GatedResidualConfig& config) -> std::optional<GatedResidualConfig> {
                return createOptimalConfigCommon(config,
                                                 dnnlGatedResidualTypeMapping,
                                                 std::vector<LayoutType>(7, LayoutType::ncsp),
                                                 gatedResidualMappingNotation);
            },
            AcceptsAnyShape<GatedResidualAttrs>,
            CreateDefault<GatedResidualDnnlExecutor, GatedResidualAttrs>{}
        )
    };

    return gatedResidualImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
