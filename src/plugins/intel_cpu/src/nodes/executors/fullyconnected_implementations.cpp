// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpp/maybe_unused.hpp"
#include "utils/debug_capabilities.h"

#if defined(OV_CPU_WITH_KLEIDIAI)
#    include "nodes/executors/kleidiai/kleidiai_mm.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_fullyconnected.hpp"
#    include "nodes/executors/acl/acl_lowp_fullyconnected.hpp"
#endif

#if defined(OV_CPU_WITH_SHL)
#    include "nodes/executors/shl/shl_fullyconnected.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

static const MappingNotation dnnlFCMappingNotation{ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST};

using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig dnnlFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig aclFCLayoutConfig{LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp};

template <dnnl::impl::cpu::x64::cpu_isa_t ISA>
struct Require {
    bool operator()() {
        return dnnl::impl::cpu::x64::mayiuse(ISA);
    }
};

// clang-format off
static const TypeMapping dnnlFCTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16, _f16, _any, _f16 | _f32},                         pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any | _any},                  pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    // int8 inner_product does not support f16 output and bias
    {{_u8 | _i8, _i8, _u8 | _i8 | _i32 | _bf16 | _f32 | _dynamic, _u8 | _i8 | _i32 | _bf16 | _f32}, pt(bypass(), bypass(), bypass(),  bypass())},
    {{_u8 | _i8, _i8, _f16, _u8 | _i8 | _i32 | _bf16 | _f32}, pt(bypass(), bypass(), just<f32>(), bypass())},
    {{_u8 | _i8, _i8, _any, _any}, pt(bypass(), bypass(), just<f32>(), just<f32>())},
    // compresses int weights (@todo more strict requrements for output precision?)
    {{_bf16, _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1, _any, _any},       pt(bypass(), bypass(), use<0>(), use<0>()),
     Require<dnnl::impl::cpu::x64::avx512_core_bf16>()}, // Ticket 122347
    {{_bf16, _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1, _any, _any},       pt(just<f32>(), bypass(), just<f32>(), just<f32>())},
    {{_f32,  _u8 | _i8 | _nf4 | _u4 | _i4 | _f4e2m1, _any, _any},       pt(bypass(), bypass(), use<0>(), use<0>())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
    // @todo explicitly cover configuration limitations for oneDNN on ARM
};

static const TypeMapping aclFCTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_f32 | _f16, _f32 | _f16, _any, _any}, pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_any, _any, _any, _any},               pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())}
};

static const TypeMapping aclLowpFCTypeMapping {
    // {src, wei, bia, dst}                  pt<src, wei, bias, dst>
    {{_i8, _i8, _any, _f32},                 pt(bypass(), bypass(), use<3>(), bypass())}
};

static const MappingNotation dnnlConvolutionMappingNotation {
    ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST
};

static const MappingNotation aclFullyConnectedMappingNotation {
    ARG_SRC, ARG_WEI, ARG_BIAS, ARG_DST
};

static const TypeMapping dnnlConvolutionTypeMapping {
    // {src, wei, bia, dst}                        pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},    pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16, _f16, _any, _f16 | _f32},              pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8}, pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any | _any},       pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},             pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},             pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    {{_u8 | _i8, _i8, _any, _any},                 pt(bypass(), bypass(), use<3>(), bypass())},
    // @todo should we fallback to _fxx instead of _f32 (currenly legacy logic is replicated)
    {{_any, _any, _any, _any},                     pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
};

static const TypeMapping dnnlMatMulTypeMapping {
    // {src, wei, bia, dst}                                   pt<src, wei, bias, dst>
    {{_bf16, _bf16 | _f32, _any, _bf16 | _f32},               pt(bypass(), bypass(), use<3>(), bypass())},
    {{_f16, _f16, _any, _f16 | _f32},                         pt(bypass(), bypass(), use<3>(), bypass())},
    // integer precision outputs are not supported for float precision inputs
    {{_f32 | _bf16 | _f16, _any, _any, _i8 | _u8},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // compresses float weights which do not match input data precision
    {{_f32, _half_float, _any, _any | _any},                  pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_bf16, _f16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    {{_f16, _bf16, _any, _any | _any},                        pt(bypass(), bypass(), use<0>(), use<0>())},
    // quantization configuration
    {{_u8 | _i8, _i8, _u8|_i8|_i32|_bf16|_f16|_f32|_dynamic, _u8|_i8|_i32|_bf16|_f16|_f32}, pt(bypass(), bypass(), bypass(),  bypass())},
    {{_u8 | _i8, _i8, _any, _any},                            pt(bypass(), bypass(), just<f32>(), just<f32>())},
    // compresses int weights
    {{_f32 | _bf16 | _f16, _u8 | _i8, _any, _any},            pt(bypass(), bypass(), use<0>(), use<0>())},
    // @todo should we fallback to FPXX instead of _f32?
    {{_any, _any, _any, _any},                                pt(just<f32>(), just<f32>(), just<f32>(), just<f32>())},
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

        if (desc->empty()) {
            continue;
        }

        if (desc->getPrecision() != type) {
            return false;  // type mismatch
        }

        if (!desc->hasLayoutType(layoutConfig[i])) {
            return false;  // layout mismatch
        }
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

        if (desc->empty()) {
            continue;
        }

        if (descType == type && desc->hasLayoutType(layout)) {
            continue;
        }

        descs[notation[i]] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
    }

    return descs;
}

template <typename Attrs>
std::optional<executor::Config<Attrs>> requiresFallbackCommon(const executor::Config<Attrs>& config,
                                                              const TypeMapping& typeMapping,
                                                              const LayoutConfig& layoutConfig,
                                                              const MappingNotation& notation) {
    const auto typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return std::optional<executor::Config<Attrs>>(FCConfig{optimalDescriptors, config.attrs, config.postOps});
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noWeightsDecompression(const FCConfig& config) {
    return !DnnlFCPrimitive::useWeightsDecompressionImpl(srcType(config), weiType(config), config.attrs.modelType);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noSparseDecompression(const FCConfig& config) {
    return !(config.attrs.sparseWeights);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noPostOps(const FCConfig& config) {
    return config.postOps.empty();
}

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<FCAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<FCAttrs>> fullyconnectedImplementations {
        OV_CPU_INSTANCE_MLAS_X64(
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
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
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
               const ExecutorContext::CPtr& context) {
                return std::make_shared<MlasGemmExecutor>(attrs, postOps, memory, context);
            })
        OV_CPU_INSTANCE_X64(
            "convolution_1x1_dnnl",
            ExecutorType::Dnnl,
            OperationType::Convolution,
            ShapeTolerance::Dependant,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                auto getOffset0 = [](const MemoryDescPtr& desc) {
                    DnnlMemoryDescCPtr dnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(desc);
                    dnnl::impl::memory_desc_wrapper wrapped(dnnlDesc->getDnnlDesc().get());
                    return wrapped.offset0();
                };

                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);
                VERIFY(srcType(config) == ov::element::f32, UNSUPPORTED_SRC_PRECISIONS);
                // disable rank=4:
                // if layout is nhwc:
                //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
                //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
                // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
                //   problems with the above.
                VERIFY(one_of(srcRank(config), 2u, 3u), UNSUPPORTED_SRC_RANK);
                VERIFY(weiRank(config) == 2, UNSUPPORTED_WEI_RANK);
                // brg convolution does not support stride
                VERIFY(getOffset0(config.descs.at(ARG_DST)) == 0, UNSUPPORTED_DST_STRIDES);
                return true;
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                // @todo use dnnlConvolutionLayoutConfig after one is implemented
                return requiresFallbackCommon(config,
                                              dnnlConvolutionTypeMapping,
                                              dnnlFCLayoutConfig,
                                              dnnlFCMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                const auto inRank = memory.at(ARG_SRC)->getShape().getRank();
                const auto& inDims = memory.at(ARG_SRC)->getShape().getDims();
                const auto& weightDims = memory.at(ARG_WEI)->getShape().getDims();
                // for original inner product semantics:
                //  when input is 2D tensor -> M in oneDNN will map to widthInConv
                //  when input is 3D tensor -> M in oneDNN will map to widthInConv*minibatch
                // currently nwc mapping in brg::
                //  when input is 2D tensor -> widthInConv will map to 'w', 'n' will be 1
                //  when input is 3D tensor -> widthInConv will map to 'w', 'n' will be minibatch
                Dim widthInConv = inDims[inRank - 2];
                Dim K = inDims[inRank - 1];
                Dim N = weightDims[0];

                const auto& weightsSize = memory.at(ARG_WEI)->getDesc().getCurrentMemSize();
                // Disable Conv1x1 when weight size >= 16M to avoid different weight layout when having different input
                // activation shapes. As a consuquence, peak memory consumption in LLM can be decreased.
                VERIFY(weightsSize < (16 * 1 << 20), " weights size is to big");
                VERIFY(widthInConv >= 2 && widthInConv <= 3136 && K >= 96 && K <= 4096 && N >= 96 && N <= K * 4,
                       HEURISTICS_MISMATCH);

                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                struct ConvolutionInstantiator {
                    std::shared_ptr<DnnlConvolutionPrimitive> operator()(
                        const MemoryArgs& memory,
                        const FCAttrs& attrs,
                        const ExecutorContext::CPtr& context,
                        const std::shared_ptr<DnnlShapeAgnosticData>& shareAgnosticData) const {
                        ConvAttrs convAttrs{attrs.withBias};
                        auto primitive =
                            DefaultInstantiator<DnnlConvolutionPrimitive, ConvAttrs, DnnlShapeAgnosticData>{}(
                            memory,
                            convAttrs,
                            context,
                            shareAgnosticData);

                        if (!primitive || primitive->implType() != brgconv_avx512_1x1) {
                            // only brgconv_avx512_1x1 primitive is acceptable from the performance perspective
                            return nullptr;
                        }
                        return primitive;
                    }
                };

                return std::make_shared<
                    DnnlFCExecutor<DnnlConvolutionPrimitive, FCAttrs, DnnlShapeAgnosticData, ConvolutionInstantiator>>(
                    attrs,
                    postOps,
                    memory,
                    context,
                    false);
            })
        OV_CPU_INSTANCE_ACL(
            "fullyconnected_acl",
            ExecutorType::Acl,
            OperationType::FullyConnected,
            ShapeTolerance::Agnostic,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                return ACLFullyConnectedExecutor::supports(config);
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return requiresFallbackCommon(config,
                                              aclFCTypeMapping,
                                              aclFCLayoutConfig,
                                              aclFullyConnectedMappingNotation);
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
               const ExecutorContext::CPtr& context) {
                return std::make_shared<ACLFullyConnectedExecutor>(attrs, postOps, memory, context);
            })
        OV_CPU_INSTANCE_ACL(
            "fullyconnected_acl_lowp",
            ExecutorType::Acl,
            OperationType::FullyConnected,
            ShapeTolerance::Agnostic,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                return ACLLowpFullyConnectedExecutor::supports(config);
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return requiresFallbackCommon(config,
                                              aclLowpFCTypeMapping,
                                              aclFCLayoutConfig,
                                              aclFullyConnectedMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                const auto dequantizationScales = getDeQuantizedScales(memory);
                bool isPerChannelQuantization = dequantizationScales.size() > 1;
                // per-channel quantization is not unsupported by ACL
                return !isPerChannelQuantization;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) {
                return std::make_shared<ACLLowpFullyConnectedExecutor>(attrs, postOps, memory, context);
            })
        OV_CPU_INSTANCE_KLEIDIAI(
            "fullyconnected_kleidiai",
            ExecutorType::Kleidiai,
            OperationType::MatMul,
            ShapeTolerance::Agnostic,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noPostOps(config), UNSUPPORTED_POST_OPS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(everyone_is(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);
                if (config.attrs.withBias) {
                    VERIFY(biaType(config) == f32, UNSUPPORTED_SRC_PRECISIONS);
                }
                VERIFY(srcRank(config) == 2U, UNSUPPORTED_SRC_RANK);
                VERIFY(weiRank(config) == 2U, UNSUPPORTED_WEI_RANK);
                return MatMulKleidiAIExecutor::supports(config);
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return {};
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const FCAttrs& attrs, const PostOps& postOps, const MemoryArgs& memory, ExecutorContext::CPtr context) {
                return std::make_shared<MatMulKleidiAIExecutor>(attrs, postOps, memory, context);
            })
        OV_CPU_INSTANCE_SHL(
            "fullyconnected_shl",
            ExecutorType::Shl,
            OperationType::FullyConnected,
            ShapeTolerance::Agnostic,
            // supports
            [](const FCConfig& config) -> bool {
                VERIFY(noPostOps(config), UNSUPPORTED_POST_OPS);
                VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                VERIFY(noWeightsDecompression(config), UNSUPPORTED_WEIGHTS_DECOMPRESSION);
                VERIFY(everyone_is(f32, srcType(config), weiType(config), dstType(config)), UNSUPPORTED_SRC_PRECISIONS);

                return ShlFCExecutor::supports(config);
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return {};
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) {
                return std::make_shared<ShlFCExecutor>(attrs, postOps, memory, context);
            }
        )
        OV_CPU_INSTANCE_DNNL(
            "matmul_dnnl",
            ExecutorType::Dnnl,
            OperationType::MatMul,
            ShapeTolerance::Dependant,
            // supports
            [](const FCConfig& config) -> bool {
                // enable only with debug caps and env variable defined for now
                CPU_DEBUG_CAP_ENABLE(
                    if (getEnvBool("OV_CPU_ENABLE_DNNL_MAMTUL_FOR_FC")) {
                        VERIFY(noSparseDecompression(config), UNSUPPORTED_SPARSE_WEIGHTS);
                        return true;
                    })
                return false;
            },
            // requiresFallback
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return requiresFallbackCommon(config,
                                              dnnlMatMulTypeMapping,
                                              dnnlFCLayoutConfig,
                                              dnnlFCMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                struct MatMulInstantiator {
                    std::shared_ptr<DnnlMatMulPrimitive> operator()(
                        const MemoryArgs& memory,
                        const FCAttrs& attrs,
                        const ExecutorContext::CPtr& context,
                        const std::shared_ptr<DnnlShapeAgnosticData>& shareAgnosticData) const {
                        MatMulAttrs matMulAttrs{false,
                                                false};
                        auto primitive =
                            DefaultInstantiator<DnnlMatMulPrimitive, MatMulAttrs, DnnlShapeAgnosticData>{}(
                            memory,
                            matMulAttrs,
                            context,
                            shareAgnosticData);
                        return primitive;
                    }
                };

                return std::make_shared<
                    DnnlFCExecutor<DnnlMatMulPrimitive, FCAttrs, DnnlShapeAgnosticData, MatMulInstantiator>>(
                    attrs,
                    postOps,
                    memory,
                    context,
                    false);
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
            [](const FCConfig& config) -> std::optional<executor::Config<FCAttrs>> {
                return requiresFallbackCommon(config,
                                              dnnlFCTypeMapping,
                                              dnnlFCLayoutConfig,
                                              dnnlConvolutionMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                return true;
            },
            // create
            [](const FCAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr& context) {
                return std::make_shared<DnnlFCExecutor<DnnlFCPrimitive, FCAttrs, DnnlShapeAgnosticData>>(attrs,
                                                                                                         postOps,
                                                                                                         memory,
                                                                                                         context,
                                                                                                         false);
            })
    };

    return fullyconnectedImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu
