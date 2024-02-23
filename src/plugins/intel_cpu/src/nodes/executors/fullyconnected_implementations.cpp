// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "openvino/core/type/element_type.hpp"
#include "ov_optional.hpp"
#include "utils/cpp/maybe_unused.hpp"

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

static const MappingNotation dnnlFCMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

// clang-format off
static const TypeMapping dnnlFCTypeMapping {
    // {src, wei, bia, dst}                              pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},          pt(bypass(), bypass(), use<3>(), use<3>())},
    {{_f16, _f16, _any, _f16 | _f32},                    pt(bypass(), bypass(), use<3>(), use<3>())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},       pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any | _any},             pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},                   pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},                   pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration (@todo more strict requrements for output precision?)
    {{_u8 | _i8, _i8, _any, _any},                       pt(bypass(), bypass(), bypass(), use<3>())},
    // compresses int weights (@todo more strict requrements for output precision?)
    {{_f32 | _bf16, _u8 | _nf4 | _u4 | _i4, _any, _any}, pt(bypass(), bypass(), use<0>(), use<0>())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                           pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};
// clang-format on

static bool fullyMatchConfiguration(const MemoryDescArgs& currentDescriptors,
                                    const InOutTypes& typeConfig,
                                    const LayoutConfig& layoutConfig,
                                    const MappingNotation& notation) {
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& type = typeConfig[i];
        const auto& desc = currentDescriptors.at(notation[i]);
        if ((!one_of(desc->getPrecision(), type, ov::element::undefined)) || !desc->hasLayoutType(layoutConfig[i]))
            return false;
    }

    return true;
}

static MemoryDescArgs createOptimalDescriptors(const MemoryDescArgs& currentDescriptors,
                                               const InOutTypes& typeConfig,
                                               const LayoutConfig& layoutConfig,
                                               const MappingNotation& notation) {
    MemoryDescArgs descs = currentDescriptors;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& desc = currentDescriptors.at(notation[i]);
        const auto& descType = desc->getPrecision();
        const auto& type = typeConfig[i];
        const auto& layout = layoutConfig[i];

        if (one_of(descType, ov::element::undefined, type)) {
            continue;
        }

        descs[notation[i]] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
    }

    return descs;
}

template <typename Attrs>
ov::optional<executor::Config<Attrs>> requiresFallbackCommon(const executor::Config<Attrs>& config,
                                                             const TypeMapping& typeMapping,
                                                             const LayoutConfig& layoutConfig,
                                                             const MappingNotation& notation) {
    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return ov::optional<executor::Config<Attrs>>(FCConfig{optimalDescriptors, config.attrs, config.postOps});
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noWeightsDecompression(const FCConfig& config) {
    return !DnnlFCPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config));
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noSparseDecompression(const FCConfig& config) {
    return !(config.attrs.sparseWeights);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noPostOps(const FCConfig& config) {
    return config.postOps.empty();
}

template <>
const std::vector<ExecutorImplementation<FCAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<FCAttrs>> fullyconnectedImplementations {
        OV_CPU_INSTANCE_X64(
            "fullyconnected_mlas",
            ExecutorType::Mlas,
            OperationType::MatMul,
            ShapeTolerance::Agnostic,
            // supports
            [](const FCConfig& config) -> bool {
                // @todo probably there is no need of having implementation name in the debug message
                // since it can be distinguished from the context of other logs anyway.
                VERIFY(noPostOps(config), UNSUPPORTED_POST_OPS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(everyone_is(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);

                return MlasGemmExecutor::supports(config);
            },
            // requiresFallback
            [](const FCConfig& config) -> ov::optional<executor::Config<FCAttrs>> {
                // @todo Implement proper handling for the cases when fallback is not expected
                // throwing exception is not an option, since requiresFallback is used in two contexts:
                // 1) getting proper memory descriptors configuration
                // 2) actual fallback to subgraph
                return {};
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<MlasGemmExecutor>(attrs, postOps, memory, context);
            })
        OV_CPU_INSTANCE_DNNL(
            "fullyconnected_dnnl",
            ExecutorType::Dnnl,
            OperationType::FullyConnected,
            ShapeTolerance::Dependant,
            // supports
            [](const FCConfig& config) -> bool {
                return true;
            },
            // requiresFallback
            [](const FCConfig& config) -> ov::optional<executor::Config<FCAttrs>> {
                return requiresFallbackCommon(config,
                                              dnnlFCTypeMapping,
                                              dnnlFCLayoutConfig,
                                              dnnlFCMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const FCAttrs& attrs, const PostOps& postOps, const MemoryArgs& memory, ExecutorContext::CPtr context) {
                return std::make_shared<DnnlFCExecutor<DnnlFCPrimitive, FCAttrs, DnnlShapeAgnosticData>>(attrs,
                                                                                                         postOps,
                                                                                                         memory,
                                                                                                         context,
                                                                                                         false);
            })
    };

    return fullyconnectedImplementations;
}
}  // namespace intel_cpu
}  // namespace ov
