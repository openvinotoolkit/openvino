// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "debug_messages.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_conv.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

static const MappingNotation dnnlConvolutionMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};
static const LayoutConfig aclConvLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

// clang-format off
static const TypeMapping dnnlConvTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f16 | _f32, _any, _bf16 | _f32},        pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16,  _bf16 | _f16 | _f32, _any, _f16  | _f32},        pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float | _i8, _any, _any},            pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    {{_u8 | _i8, _i8, _quant | _hw_float | _i32 | _dynamic, _quant | _hw_float | _i32 | _dynamic}, pt(bypass(), bypass(), bypass(),  bypass())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};

static const TypeMapping aclLowpConvTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_u8, _u8 | _i8, _i32 | _dynamic, _u8 | _f32},           pt(bypass(), bypass(), bypass(), bypass())},
    {{_i8, _i8, _i32 | _dynamic, _i8 | _f32},                 pt(bypass(), bypass(), bypass(), bypass())},

};
// clang-format on
struct RequiresFallbackDefault {
    std::optional<ConvConfig> operator()(const ConvConfig& config) const {
        return requiresFallbackCommon(config, dnnlConvTypeMapping, layoutConfig, dnnlConvolutionMappingNotation);
    }

    LayoutConfig layoutConfig;
};

template <typename Attrs>
bool MatchesMemoryFormatFilter(const executor::Config<Attrs>& config,
                               const LayoutConfig& layoutConfig,
                               const MemoryFormatFilter& filter) {
    const auto& notation = dnnlConvolutionMappingNotation;

    for (size_t i = 0; i < filter.input.size(); i++) {
        const auto& desc = config.descs.at(notation[i]);

        if (desc->empty()) {
            continue;
        }

        const auto dnnlDesc = DnnlBlockedMemoryDesc(config.descs.at(notation[i])->getShape(),
                                                    dnnl::memory::data_type::f32,
                                                    filter.input[i]);
        if (!dnnlDesc.hasLayoutType(layoutConfig[i])) {
            return false;
        }
    }

    if (filter.output.empty()) {
        return true;
    }

    const auto desc = DnnlBlockedMemoryDesc(config.descs.at(ARG_DST)->getShape(),
                                            dnnl::memory::data_type::f32,
                                            filter.output.front());
    if (!desc.hasLayoutType(layoutConfig.back())) {
        return false;
    }

    return true;
}

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<ConvAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<ConvAttrs>> convolutionImplementations {
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }
                // nspc shows better performance only with brgconv implementation
                return DnnlConvolutionPrimitive::isBrgConvAvailable(config);
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC == 1 && groupOC == 1;
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp16c_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            RequiresFallbackDefault{{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp8c_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            RequiresFallbackDefault{{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp_unconditional", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                               memoryFormatFilter)) {
                    return false;
                }

                return true;
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc_backup", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }

                return !one_of(srcType(config), ov::element::bf16, ov::element::f16) && DnnlConvolutionPrimitive::isNspcAvailable(config);
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_dnnl_nspc_nspc_unconditional_acl", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                std::cout << "convolution_acl_normal: src: " << srcType(config).to_string() << " wei: " << weiType(config).to_string() <<
                " bia: " << biaType(config).to_string() << " dst: " << dstType(config).to_string() << std::endl;
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }
                return one_of(srcType(config), ov::element::f32, ov::element::f16);
                //return true;
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<ConvAttrs>{},
            CreateDnnlDefault<DnnlConvolutionPrimitive, ConvAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_acl_lowp", ExecutorType::Acl, OperationType::Convolution, ShapeTolerance::Agnostic,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                //VERIFY(everyone_is(i8, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);
                //VERIFY(everyone_is(i32, biaType(config)), UNSUPPORTED_SRC_PRECISIONS);
                std::cout << "convolution_acl_lowp: src: " << srcType(config).to_string() << " wei: " << weiType(config).to_string() <<
                " bia: " << biaType(config).to_string() << " dst: " << dstType(config).to_string() << std::endl;
                return ACLConvolutionExecutor::supports(config);
            },
            [](const ConvConfig& config) -> std::optional<executor::Config<ConvAttrs>> {
                return requiresFallbackCommon(config,
                                              aclLowpConvTypeMapping,
                                              aclConvLayoutConfig,
                                              dnnlConvolutionMappingNotation);
            },
            // acceptsShapes
            [](const ConvAttrs& attrs,
               const MemoryArgs& memory) -> bool {
                const auto dequantizationScales = getDeQuantizedScales(memory);
                bool isPerChannelQuantization = dequantizationScales.size() > 1;
                //TODO: per-channel quantization is not unsupported by ACL?
                return !isPerChannelQuantization;
            },
            CreateDefault<ACLConvolutionExecutor, ConvAttrs>{}
            )
    };

    return convolutionImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
