// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
#    include <optional>

#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "debug_messages.hpp"
#    include "implementation_utils.hpp"
#    include "nodes/executors/dnnl/dnnl_gathermatmul_executor.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/executor_config.hpp"
#endif
namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)

// GatherMatmul always uses plain (ncsp) layout for all four standard arguments
static const LayoutConfig dnnlGatherMatmulLayoutConfig{LayoutType::ncsp,
                                                       LayoutType::ncsp,
                                                       LayoutType::ncsp,
                                                       LayoutType::ncsp};

template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};

// clang-format off
static const TypeMapping dnnlGatherMatmulTypeMapping {
    // {src, wei, bia, dst}                                      pt<src, wei, bias, dst>
    // float precision paths
    {{_bf16, _bf16, _any, _bf16 | _f32},                        {bypass(), bypass(), use<3>(), bypass()}},
    // oneDNN inner_product does not support mixed bf16/f32 or bf16/f16: align weights precision to src
    {{_bf16, _f16 | _f32, _any, _bf16 | _f32},                 {bypass(), use<0>(), use<3>(), bypass()}},
    {{_f32,  _f32,         _any, _f32},                         {bypass(), bypass(), use<3>(), bypass()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},                           {bypass(), bypass(), use<0>(), use<0>()}},
    // compressed int weights with float activations
    {{_f32,  _u8 | _i8 | _u4 | _i4, _any, _any},                {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _u8 | _i8 | _u4 | _i4, _any, _any},                {bypass(), bypass(), use<0>(), use<0>()},
     Require<dnnl::impl::cpu::x64::avx512_core_bf16>()},
    {{_bf16, _u8 | _i8 | _u4 | _i4, _any, _any},                {just<f32>(), bypass(), just<f32>(), just<f32>()}},
    // fallback
    {{_any,  _any, _any, _any},                                 {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
};
// clang-format on

static const MappingNotation gatherMatmulMappingNotation{
    {ARG_SRC, 0},
    {ARG_WEI, 1},
    {ARG_BIAS, 2},
    {ARG_DST, 3},
};

#endif

// clang-format off
template <>
const std::vector<ExecutorImplementation<GatherMatmulAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<GatherMatmulAttrs>> gathermatmulImplementations{
        OV_CPU_INSTANCE_DNNL_X64(
            "gathermatmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::GatherMatmul,
            // supports
            [](const GatherMatmulConfig& config) -> bool {
                VERIFY(GatherMatmulDnnlExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);
                return true;
            },
            // createOptimalConfig
            [](const GatherMatmulConfig& config) -> std::optional<executor::Config<GatherMatmulAttrs>> {
                return createOptimalConfigCommon(config,
                                                 dnnlGatherMatmulTypeMapping,
                                                 dnnlGatherMatmulLayoutConfig,
                                                 gatherMatmulMappingNotation);
            },
            AcceptsAnyShape<GatherMatmulAttrs>,
            CreateDefault<GatherMatmulDnnlExecutor, GatherMatmulAttrs>{}
        )
    };
    return gathermatmulImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
