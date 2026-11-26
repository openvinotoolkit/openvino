// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/grouped_matmul_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)
#    include <optional>

#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "debug_messages.hpp"
#    include "implementation_utils.hpp"
#    include "memory_desc/cpu_memory_desc.h"
#    include "nodes/executors/dnnl/dnnl_grouped_matmul_executor.hpp"
#    include "nodes/executors/executor.hpp"
#    include "nodes/executors/executor_config.hpp"
#    include "nodes/executors/memory_arguments.hpp"
#    include "nodes/executors/precision_translation.hpp"
#    include "nodes/executors/type_mask.hpp"
#    include "openvino/core/type/element_type.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;

#if defined(OV_CPU_WITH_DNNL) && defined(OPENVINO_ARCH_X86_64)

using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// GroupedMatMul always uses plain (ncsp) layout. ARG_GMM_OFFSETS is the optional offsets input of the
// 2D x 3D form. The empty bias argument used by the shared inner-product executor is not mapped.
static const LayoutConfig dnnlGroupedMatMulLayoutConfig{LayoutType::ncsp,
                                                        LayoutType::ncsp,
                                                        LayoutType::ncsp,
                                                        LayoutType::ncsp};

namespace {
template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};
}  // namespace

// clang-format off
static const uint64_t compressed_weights = utils::supported_compressed_weights_mask();
static const TypeMapping dnnlGroupedMatMulTypeMapping {
    // {src, wei, offsets, dst}                                  pt<src, wei, offsets, dst>
    // float precision paths
    {{_bf16, _bf16, _any, _bf16 | _f32},                        {bypass(), bypass(), just<i32>(), bypass()}},
    // oneDNN inner_product does not support mixed bf16/f32 or bf16/f16: align weights precision to src
    {{_bf16, _f16 | _f32, _any, _bf16 | _f32},                  {bypass(), use<0>(), just<i32>(), bypass()}},
    {{_f32,  _f32,         _any, _f32},                         {bypass(), bypass(), just<i32>(), bypass()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any},                           {bypass(), bypass(), just<i32>(), use<0>()}},
    // compressed int weights with float activations
    {{_f32,  compressed_weights, _any, _any},          {bypass(), bypass(), just<i32>(), use<0>()}},
    {{_bf16, compressed_weights, _any, _any},          {bypass(), bypass(), just<i32>(), use<0>()},
     Require<dnnl::impl::cpu::x64::avx512_core_bf16>()},
    {{_bf16, compressed_weights, _any, _any},          {just<f32>(), bypass(), just<i32>(), just<f32>()}},
    // fallback
    {{_any,  _any, _any, _any},                                 {just<f32>(), just<f32>(), just<i32>(), just<f32>()}},
};
// clang-format on

static const MappingNotation groupedMatMulMappingNotation{
    {ARG_GMM_DATA, 0},
    {ARG_GMM_WEIGHTS, 1},
    {ARG_GMM_OFFSETS, 2},
    {ARG_DST, 3},
};

#endif

// clang-format off
template <>
const std::vector<ExecutorImplementation<GroupedMatMulAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<GroupedMatMulAttrs>> grouped_matmul_implementations {

        OV_CPU_INSTANCE_DNNL_X64(
            "groupedmatmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::GroupedMatMul,
            // supports
            [](const GroupedMatMulConfig& config) -> bool {
                VERIFY(GroupedMatMulDnnlExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);
                return true;
            },
            // createOptimalConfig
            [](const GroupedMatMulConfig& config) -> std::optional<executor::Config<GroupedMatMulAttrs>> {
                return createOptimalConfigCommon(config,
                                                 dnnlGroupedMatMulTypeMapping,
                                                 dnnlGroupedMatMulLayoutConfig,
                                                 groupedMatMulMappingNotation);
            },
            AcceptsAnyShape<GroupedMatMulAttrs>,
            CreateDefault<GroupedMatMulDnnlExecutor, GroupedMatMulAttrs>{}
        )
    };

    return grouped_matmul_implementations;
}
// clang-format on

}  // namespace ov::intel_cpu
