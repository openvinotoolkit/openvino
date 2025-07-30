// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/general_utils.h"

#if !defined(OPENVINO_ARCH_RISCV64)
#    include "memory_format_filter.hpp"
#    include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64) || defined(OV_CPU_WITH_ACL)
#    include "cpu/x64/cpu_isa_traits.hpp"
#    include "nodes/executors/debug_messages.hpp"
#    include "nodes/executors/executor.hpp"
#    include "post_ops.hpp"
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
// clang-format on
struct CreateOptimalConfigDefault {
    std::optional<ConvConfig> operator()(const ConvConfig& config) const {
        return createOptimalConfigCommon(config, dnnlConvTypeMapping, layoutConfig, dnnlConvolutionMappingNotation);
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
            "convolution_dnnl_nspc_nspc", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
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
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                
                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return all_of(1U, IC, groupOC);
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);

                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);


                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            CreateOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp16c_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);

                // fork kernel with dw conv post ops supports only src: (ncsp | nCsp8c), dst: nCsp8c
                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                VERIFY(!hasPostOp<DepthwiseConvolutionPostOp>(config.attrs.postOps), UNSUPPORTED_POST_OPS);

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            CreateOptimalConfigDefault{{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp8c_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);
                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            CreateOptimalConfigDefault{{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp_unconditional", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
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
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc_backup", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(!isQuantized(config), UNSUPPORTED_SRC_PRECISIONS);

                return none_of(srcType(config), ov::element::bf16, ov::element::f16) && DnnlConvolutionPrimitive::isNspcAvailable(config);
            },
            CreateOptimalConfigDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_dnnl_nspc_nspc_unconditional_acl", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, dnnlConvolutionMappingNotation), MEMORY_FORMAT_MISMATCH);

                return true;
            },
            CreateOptimalConfigDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
    };

    return convolutionImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
