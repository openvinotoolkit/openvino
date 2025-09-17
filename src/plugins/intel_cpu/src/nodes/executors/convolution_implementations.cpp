// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "post_ops.hpp"
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/executor.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_conv.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// clang-format off
static const MappingNotation dnnlConvolutionMappingNotation {
    {ARG_SRC,  0},
    {ARG_WEI,  1},
    {ARG_BIAS, 2},
    {ARG_DST,  3}
};

static const TypeMapping dnnlConvTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f16 | _f32, _any, _bf16 | _f32},        {bypass(), bypass(), use<3>(), bypass()}},
    {{_f16,  _bf16 | _f16 | _f32, _any, _f16  | _f32},        {bypass(), bypass(), use<3>(), bypass()}},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            {bypass(), bypass(), use<0>(), use<0>()}},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float | _i8, _any, _any},                   {bypass(), bypass(), use<0>(), use<0>()}},
    {{_bf16, _f16, _any, _any},                               {bypass(), bypass(), use<0>(), use<0>()}},
    {{_f16, _bf16, _any, _any},                               {bypass(), bypass(), use<0>(), use<0>()}},
    // quantization configuration
    // int8 conv does not support f16 output and bias
    {{_u8 | _i8, _i8,  _quant |_bf16 | _f32 | _i32 | _dynamic,  _quant | _bf16 | _f32 | _i32 | _dynamic}, {bypass(), bypass(), bypass(),  bypass()}},
    {{_u8 | _i8, _i8, _f16, _u8 | _i8 | _i32 | _bf16 | _f32}, {bypass(), bypass(), just<f32>(), bypass()}},
    {{_u8 | _i8, _i8, _any, _any}, {bypass(), bypass(), just<f32>(), just<f32>()}},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                {just<f32>(), just<f32>(), just<f32>(), just<f32>()}},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};

static const TypeMapping aclLowpConvTypeMapping {
    // {src, wei, bia, dst}                            pt<src, wei, bias, dst>
    {{_u8, _u8 | _i8, _i32 | _dynamic, _u8},           {bypass(), bypass(), bypass(), bypass()}},
    {{_i8, _i8, _i32 | _dynamic, _i8},                 {bypass(), bypass(), bypass(), bypass()}},
};
// clang-format on
struct CreateOptimalConfigDefault {
    std::optional<ConvConfig> operator()(const ConvConfig& config) const {
        return createOptimalConfigCommon(config, dnnlConvTypeMapping, layoutConfig, dnnlConvolutionMappingNotation);
    }

    LayoutConfig layoutConfig;
};

struct CreateOptimalConfigAclLowp {
    std::optional<ConvConfig> operator()(const ConvConfig& config) const {
        return createOptimalConfigCommon(config, aclLowpConvTypeMapping, layoutConfig, dnnlConvolutionMappingNotation);
    }

    LayoutConfig layoutConfig;
};

[[maybe_unused]] static inline bool isQuantized(const ConvConfig& config) {
    return any_of(config.descs.at(ARG_SRC)->getPrecision(), ov::element::u8, ov::element::i8) &&
           config.descs.at(ARG_WEI)->getPrecision() == ov::element::i8;
};

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<ConvAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<ConvAttrs>> convolutionImplementations {
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);
                const bool is_quantized = isQuantized(config);
                const bool brg_conv_available = DnnlConvolutionPrimitive::isBrgConvAvailable(config);
                const bool valid_config = is_quantized || brg_conv_available;
                VERIFY(valid_config, "is not quantized or brgemm convolution is not available");

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                
                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);
                VERIFY(all_of(1U, IC, groupOC), HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);

                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);
                VERIFY(IC < 4 && groupOC != 1, HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);
                VERIFY(IC < 4 && groupOC != 1, HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp16c_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);

                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);
                VERIFY(IC > 4, HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp8c_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);
                VERIFY(IC > 4, HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp_unconditional", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc_backup", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(none_of(srcType(config), ov::element::bf16, ov::element::f16), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(DnnlConvolutionPrimitive::isNspcAvailable(config), HEURISTICS_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_dnnl_nspc_nspc_unconditional_acl", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_RISCV64(
            "convolution_dnnl_ref_ncsp", ExecutorType::Dnnl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(config.attrs.postOps.empty(), UNSUPPORTED_POST_OPS);
                return MatchesMemoryFormatFilter(config.descs,
                                                 LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter,
                                                 dnnlConvolutionMappingNotation);
            },
            // createOptimalConfig
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_acl_lowp", ExecutorType::Acl, OperationType::Convolution,
            // supports
            [](const ConvConfig& config, [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                return true;
            },
            CreateOptimalConfigAclLowp{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>,
            CreateDefault<ACLConvolutionExecutor, ConvAttrs>{}
            )
    };

    return convolutionImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
