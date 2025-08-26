// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

#include "cpu_shape.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_stateful_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/general_utils.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_eltwise.hpp"
#endif

#if defined(OV_CPU_WITH_SHL)
#    include "nodes/executors/shl/shl_eltwise.hpp"
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/cpu_isa_traits.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace executor;

static bool isBitwiseAlgorithm(const EltwiseConfig& config) {
    const auto algorithm = config.attrs.data.algo;
    return any_of(algorithm,
                  Algorithm::EltwiseBitwiseAnd,
                  Algorithm::EltwiseBitwiseNot,
                  Algorithm::EltwiseBitwiseOr,
                  Algorithm::EltwiseBitwiseXor,
                  Algorithm::EltwiseBitwiseLeftShift,
                  Algorithm::EltwiseBitwiseRightShift);
}

[[maybe_unused]] static bool isChannelFirstApplicable(const EltwiseConfig& config) {
    auto acceptableRank = [](const size_t rank) {
        return any_of(rank, 1U, 2U, 3U, 4U, 5U);
    };

    const auto outputRank = dstRank(config);

    if (!acceptableRank(outputRank)) {
        return false;
    }

    return std::all_of(config.descs.begin(), config.descs.end(), [&](const auto& entry) {
        const auto& [argId, desc] = entry;
        if (argId == ARG_DST) {
            return true;  // skip destination descriptor check
        }

        const auto inputRank = desc->getShape().getRank();
        if (!acceptableRank(inputRank)) {
            return false;
        }

        return implication(inputRank != 1, inputRank == outputRank);
    });
}

[[maybe_unused]] static bool isBlockedApplicable(const EltwiseConfig& config) {
    auto acceptableRank = [](const size_t rank) {
        return any_of(rank, 1U, 3U, 4U, 5U);
    };

    const auto outputRank = dstRank(config);

    if (!acceptableRank(outputRank)) {
        return false;
    }

    return std::all_of(config.descs.begin(), config.descs.end(), [&](const auto& entry) {
        const auto& [argId, desc] = entry;
        if (argId == ARG_DST) {
            return true;  // skip destination descriptor check
        }

        const auto inputRank = desc->getShape().getRank();
        if (!acceptableRank(inputRank)) {
            return false;
        }

        if (!implication(inputRank != 1, inputRank == outputRank)) {
            return false;
        }

        // check if channel dimension is > 1
        if (inputRank > 1 && desc->getShape().isDynamic()) {
            const auto& minDims = desc->getShape().getMinDims();
            if (minDims.size() > 1 && (minDims[1] == Shape::UNDEFINED_DIM || minDims[1] <= 1)) {
                return false;
            }
        }

        return true;
    });
}

using LayoutConfig = std::vector<LayoutType>;

using namespace TypeMaskAlias;

static const MappingNotation eltwiseMappingNotation{{ARG_SRC, 0},
                                                    {ARG_SRC_1, 0},
                                                    {ARG_SRC_2, 0},
                                                    {ARG_SRC_3, 0},
                                                    {ARG_SRC_4, 0},
                                                    {ARG_SRC_5, 0},
                                                    {ARG_SRC_6, 0},
                                                    {ARG_SRC_7, 0},
                                                    {ARG_SRC_8, 0},
                                                    {ARG_SRC_9, 0},
                                                    // Support for more than 10 inputs can be added if necessary
                                                    {ARG_DST, 1}};

// clang-format off
static const TypeMapping eltwiseTypeMapping {
    // {src, dst}         pt<src, dst>
    {{_any, _any},        {bypass(), bypass()}},
};

static const TypeMapping eltwiseReferenceTypeMapping {
    // {src, dst}         pt<src, dst>
    {{_any, _any},        {just<f32>(), just<f32>()}},
};

static const TypeMapping bitwiseReferenceTypeMapping {
    // {src, dst}         pt<src, dst>
    {{_u8 | _i8 | _u16 | _i16 | _i32, _u8 | _i8 | _u16 | _i16 | _i32},        {bypass(), bypass()}},
    {{_any, _any}, {just<f32>(), just<f32>()}},
};

static const TypeMapping aclEltwiseTypeMapping {
    // {src, dst}         pt<src, dst>
    {{_more_than_two_bytes, _f16}, {just<f32>(), just<f32>()}},
    {{_any, _f16},                 {use<1>(), use<1>()}},
    {{_any, _any},                 {just<f32>(), just<f32>()}},
};
// clang-format on

struct createOptimalConfigDefault {
    std::optional<EltwiseConfig> operator()(const EltwiseConfig& config) const {
        return createOptimalConfigCommon(config, eltwiseTypeMapping, layoutConfig, eltwiseMappingNotation);
    }

    LayoutConfig layoutConfig;
};

[[maybe_unused]] static std::optional<executor::Config<EltwiseAttrs>> createOptimalConfigEltwise(
    const executor::Config<EltwiseAttrs>& config,
    const TypeMapping& typeMapping,
    const std::vector<LayoutType>& layoutConfig,
    const MappingNotation& notation) {
    // @todo lambdas inside a template function can potentially increase binary size
    auto fullyMatchConfiguration = [](const MemoryDescArgs& currentDescriptors,
                                      const TypeOfArg& typeConfig,
                                      const std::vector<LayoutType>& layoutConfig,
                                      const MappingNotation& notation) {
        return std::all_of(currentDescriptors.begin(), currentDescriptors.end(), [&](const auto& entry) {
            const auto& [argId, desc] = entry;
            const auto type = typeConfig.at(argId);

            if (desc->empty()) {
                return true;  // empty descriptor is considered as a match
            }

            if (desc->getPrecision() != type) {
                return false;  // type mismatch
            }

            const size_t i = notation.at(argId);

            if (desc->getShape().getRank() < 2) {
                return true;
            }

            const bool blocked1C = any_of(layoutConfig[i], LayoutType::nCsp16c, LayoutType::nCsp8c) &&
                                   desc->getShape().getMinDims()[1] == 1;
            if (blocked1C) {
                return true;
            }

            return desc->hasLayoutType(layoutConfig[i]);
        });
    };

    auto createOptimalDescriptors = [](const MemoryDescArgs& currentDescriptors,
                                       const TypeOfArg& typeConfig,
                                       const std::vector<LayoutType>& layoutConfig,
                                       const MappingNotation& notation) {
        MemoryDescArgs descs = currentDescriptors;

        const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
        for (const auto& [argId, desc] : currentDescriptors) {
            if (desc->empty()) {
                continue;  // skip empty descriptors
            }

            const size_t i = notation.at(argId);
            const auto& type = typeConfig.at(argId);
            const auto& layout = layoutConfig[i];

            if (desc->getPrecision() == type && desc->hasLayoutType(layout)) {
                continue;  // already optimal
            }

            auto alwaysNCSP = [](const MemoryDescPtr& desc, LayoutType layout) {
                if (desc->getShape().getRank() < 2) {
                    return true;
                }

                const size_t channelSize = desc->getShape().getMinDims()[1];
                const bool blocked1C = any_of(layout, LayoutType::nCsp16c, LayoutType::nCsp8c) && channelSize <= 1;
                return blocked1C;
            };

            if (alwaysNCSP(desc, layout)) {
                descs[argId] = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(type, desc->getShape());
                continue;
            }

            descs[argId] = creatorsMap.at(layout)->createSharedDesc(type, desc->getShape());
        }

        return descs;
    };

    const TypeOfArg typeConfig = getTypeConfiguration(config.descs, typeMapping, notation);

    if (fullyMatchConfiguration(config.descs, typeConfig, layoutConfig, notation)) {
        return {};
    }

    const auto optimalDescriptors = createOptimalDescriptors(config.descs, typeConfig, layoutConfig, notation);

    return std::optional<executor::Config<EltwiseAttrs>>(
        executor::Config<EltwiseAttrs>{optimalDescriptors, config.attrs});
}

// clang-format off
template <>
const std::vector<ExecutorImplementation<EltwiseAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<EltwiseAttrs>> eltwiseImplementations {
        OV_CPU_INSTANCE_COMMON(
            "eltwise_jit_ncsp", ExecutorType::Jit, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(EltwiseJitExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            createOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "eltwise_jit_nspc", ExecutorType::Jit, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(isChannelFirstApplicable(config), HEURISTICS_MISMATCH);
                VERIFY(EltwiseJitExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            createOptimalConfigDefault{{LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_X64(
            "eltwise_jit_nCsp16c", ExecutorType::Jit, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);
                VERIFY(isBlockedApplicable(config), HEURISTICS_MISMATCH);
                VERIFY(EltwiseJitExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                return createOptimalConfigEltwise(config,
                                                  eltwiseTypeMapping,
                                                  {LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                  eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_X64(
            "eltwise_jit_nCsp8c", ExecutorType::Jit, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(isBlockedApplicable(config), HEURISTICS_MISMATCH);
                VERIFY(EltwiseJitExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);
                return true;
            },
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                return createOptimalConfigEltwise(config,
                                                  eltwiseTypeMapping,
                                                  {LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                  eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "eltwise_acl_ncsp", ExecutorType::Acl, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(ACLEltwiseExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            // createOptimalConfig
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                return createOptimalConfigEltwise(config,
                                                  aclEltwiseTypeMapping,
                                                  LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                  eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<ACLEltwiseExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_ACL(
            "eltwise_acl_nspc", ExecutorType::Acl, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(isChannelFirstApplicable(config), HEURISTICS_MISMATCH);
                VERIFY(ACLEltwiseExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            // createOptimalConfig
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                return createOptimalConfigEltwise(config,
                                                  aclEltwiseTypeMapping,
                                                  LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                  eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<ACLEltwiseExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_SHL(
            "eltwise_shl_ncsp", ExecutorType::Shl, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(ShlEltwiseExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            createOptimalConfigDefault{{LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<ShlEltwiseExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_SHL(
            "eltwise_shl_nspc", ExecutorType::Shl, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(ShlEltwiseExecutor::supports(config), UNSUPPORTED_BY_EXECUTOR);

                return true;
            },
            createOptimalConfigDefault{{LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<ShlEltwiseExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "eltwise_ref_ncsp", ExecutorType::Reference, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                return true;
            },
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                const auto& typeMapping = isBitwiseAlgorithm(config) ? bitwiseReferenceTypeMapping : eltwiseReferenceTypeMapping;
                return createOptimalConfigCommon(config,
                                                 typeMapping,
                                                 LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
        OV_CPU_INSTANCE_COMMON(
            "eltwise_ref_nspc", ExecutorType::Reference, OperationType::Eltwise,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                VERIFY(isChannelFirstApplicable(config), HEURISTICS_MISMATCH);
                return true;
            },
            // createOptimalConfig
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                const auto& typeMapping = isBitwiseAlgorithm(config) ? bitwiseReferenceTypeMapping : eltwiseReferenceTypeMapping;
                return createOptimalConfigCommon(config,
                                                 typeMapping,
                                                 LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>,
            CreateDefault<EltwiseStatefulExecutor, EltwiseAttrs>{}
            )
    };

    return eltwiseImplementations;
}
// clang-format on
}  // namespace ov::intel_cpu
