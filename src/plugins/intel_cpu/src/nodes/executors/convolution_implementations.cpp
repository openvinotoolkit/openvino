// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

static const MappingNotation dnnlConvolutionMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};

// clang-format off
static const TypeMapping dnnlConvTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f16 | _f32, _any, _bf16 | _f32},        pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16,  _bf16 | _f16 | _f32, _any, _f16  | _f32},        pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float | _i8, _any, _any | _any},            pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    {{_u8 | _i8, _i8, _quant | _hw_float | _i32 | _dynamic, _quant | _hw_float | _i32 | _dynamic}, pt(bypass(), bypass(), bypass(),  bypass())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};
// clang-format on

static const LayoutConfig dnnlNcspLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

static const LayoutConfig dnnlNspcLayoutConfig{LayoutType::nspc, LayoutType::nspc, LayoutType::nspc, LayoutType::nspc};

static const LayoutConfig dnnlNcsp8cLayoutConfig{LayoutType::nCsp8c,
                                                 LayoutType::nCsp8c,
                                                 LayoutType::nCsp8c,
                                                 LayoutType::nCsp8c};
static const LayoutConfig dnnlNcsp16cLayoutConfig{LayoutType::nCsp16c,
                                                  LayoutType::nCsp16c,
                                                  LayoutType::nCsp16c,
                                                  LayoutType::nCsp16c};

template <typename Attrs>
struct SupportsAnyConfig {
    bool operator()(const executor::Config<Attrs>&) const {
        return true;
    }
};

struct AcceptsAnyShape {
    bool operator()(const ConvAttrs&, const PostOps&, const MemoryArgs&) const {
        return true;
    }
};

struct NoSumBroadcast {
    bool operator()(const ConvAttrs& attrs, const PostOps& postOps, const MemoryArgs& args) const {
        if (!args.at(ARG_DST)->isDefined()) {
            return true;  // cannot decide when shapes are not defined yet
        }

        const auto sumPostOp = std::find_if(postOps.begin(), postOps.end(), [](const std::shared_ptr<PostOp>& po) {
            return std::dynamic_pointer_cast<SumPostOp>(po);
        });

        if (sumPostOp == postOps.end()) {
            return true;
        }

        const auto& sumPostOpShape = args.at(ARG_SUM)->getShape();
        const auto& dstShape = args.at(ARG_DST)->getShape();

        if (sumPostOpShape.getStaticDims() != dstShape.getStaticDims()) {
            return false;
        }

        return true;
    }
};

struct CreateDefault {
    ExecutorPtr operator()(const ConvAttrs& attrs,
                           const PostOps& postOps,
                           const MemoryArgs& memory,
                           const ExecutorContext::CPtr& context) const {
        return std::make_shared<DnnlFCExecutor<DnnlConvolutionPrimitive, ConvAttrs, DnnlShapeAgnosticData>>(attrs,
                                                                                                            postOps,
                                                                                                            memory,
                                                                                                            context,
                                                                                                            false);
    }
};

template <typename Attrs>
struct RequiresFallbackDefault {
    std::optional<executor::Config<Attrs>> operator()(const executor::Config<Attrs>& config) const {
        return requiresFallbackCommon(config, dnnlConvTypeMapping, layoutConfig, dnnlConvolutionMappingNotation);
    }

    LayoutConfig layoutConfig;
};

template <typename Attrs>
bool MatchesMemoryFormatFilter(const executor::Config<Attrs>& config,
                               const LayoutConfig& layoutConfig,
                               const MemoryFormatFilter& filter) {
    const auto notation = dnnlConvolutionMappingNotation;

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
            "convolution_dnnl_nspc_nspc", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }
                // nspc shows better performance only with brgconv implementation
                return DnnlConvolutionPrimitive::isBrgConvAvailable(config);
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC == 1 && groupOC == 1;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC < 4 && groupOC != 1;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp16c_nCsp16c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::nCsp16c, LayoutType::ncsp, LayoutType::nCsp16c, LayoutType::nCsp16c}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nCsp8c_nCsp8c", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c},
                                               memoryFormatFilter)) {
                    return false;
                }

                const auto [groupNum, groupIC, IC, groupOC] = DnnlConvolutionPrimitive::getChannelParams(config);

                return IC > 4;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::nCsp8c, LayoutType::ncsp, LayoutType::nCsp8c, LayoutType::nCsp8c}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_ncsp_ncsp_unconditional", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp},
                                               memoryFormatFilter)) {
                    return false;
                }

                return true;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_DNNL_X64(
            "convolution_dnnl_nspc_nspc_backup", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }

                return !one_of(srcType(config), ov::element::bf16, ov::element::f16) && DnnlConvolutionPrimitive::isNspcAvailable(config);
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            NoSumBroadcast{},
            CreateDefault{}
            )
        OV_CPU_INSTANCE_ACL(
            "convolution_dnnl_nspc_nspc_unconditional_acl", ExecutorType::Dnnl, OperationType::Convolution,  ShapeTolerance::Dependant,
            // supports
            [](const ConvConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                if (!MatchesMemoryFormatFilter(config, LayoutConfig{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc},
                                               memoryFormatFilter)) {
                    return false;
                }

                return true;
            },
            RequiresFallbackDefault<ConvAttrs>{{LayoutType::nspc, LayoutType::ncsp, LayoutType::nspc, LayoutType::nspc}},
            NoSumBroadcast{}, // acceptsShape
            CreateDefault{}
            )
    };

    return convolutionImplementations;
}
// clang-format on

}  // namespace intel_cpu
}  // namespace ov
