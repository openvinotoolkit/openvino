// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_format_filter.hpp"
#include "nodes/common/blocked_desc_creator.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/executors/jit/eltwise_executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_eltwise.hpp"
#endif

#if defined(OV_CPU_WITH_SHL)
#    include "nodes/executors/shl/shl_eltwise.hpp"
#endif

namespace ov::intel_cpu {

using namespace ov::element;
using namespace executor;

// static bool isBitwiseAlgorithm(const EltwiseConfig& config) {
//     const auto algorithm = config.attrs.algorithm;
//     return one_of(algorithm,
//                   Algorithm::EltwiseBitwiseAnd,
//                   Algorithm::EltwiseBitwiseNot,
//                   Algorithm::EltwiseBitwiseOr,
//                   Algorithm::EltwiseBitwiseXor,
//                   Algorithm::EltwiseBitwiseLeftShift,
//                   Algorithm::EltwiseBitwiseRightShift);
// }

static bool supportsJitExecution(const EltwiseConfig& config) {
    return EltwiseJitExecutor::isSupportedConfiguration(config.attrs, srcRank(config));
}

static bool supportsRefExecution([[maybe_unused]] const EltwiseConfig& config) {
    // Reference executor supports all operations
    return true;
}

using LayoutConfig = std::vector<LayoutType>;

// LayoutType layoutType(const ConvConfig& config, int idx) {
//     return config.descs.at(idx)->getLayoutType();
// }

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

struct RequiresFallbackDefault {
    std::optional<EltwiseConfig> operator()(const EltwiseConfig& config) const {
        return requiresFallbackCommon(config, eltwiseTypeMapping, layoutConfig, eltwiseMappingNotation);
    }

    LayoutConfig layoutConfig;
};

[[maybe_unused]] static std::optional<executor::Config<EltwiseAttrs>> requiresFallbackEltwise(
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

            if (desc->getShape().getRank() < 2 || desc->getShape().getMinDims()[1] <= 1) {
                return true;
            }

            const int i = notation.at(argId);
            if (desc->getShape().getRank() > 2 && !desc->hasLayoutType(layoutConfig[i])) {
                return false;  // layout mismatch
            }

            return true;
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

            const int i = notation.at(argId);
            const auto& type = typeConfig.at(argId);
            const auto& layout = layoutConfig[i];

            if (desc->getPrecision() == type && desc->hasLayoutType(layout)) {
                continue;  // already optimal
            }

            if (desc->getShape().getRank() < 2 || desc->getShape().getMinDims()[1] <= 1) {  // rank 1 tensors are always ncsp
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

    return std::optional<executor::Config<EltwiseAttrs>>(executor::Config<EltwiseAttrs>{optimalDescriptors, config.attrs});
}

template <>
const std::vector<ExecutorImplementation<EltwiseAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<EltwiseAttrs>> eltwiseImplementations {
        // Shape agnostic JIT executor
        // OV_CPU_INSTANCE_X64(
        OV_CPU_INSTANCE_COMMON(
            "eltwise_jit_ncsp", ExecutorType::jit_x64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);

                return supportsJitExecution(config);
            },
            // RequiredNoFallback<EltwiseAttrs>{},
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        // OV_CPU_INSTANCE_X64(
        OV_CPU_INSTANCE_COMMON(
            "eltwise_jit_nspc", ExecutorType::jit_x64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);

                return supportsJitExecution(config);
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        OV_CPU_INSTANCE_X64(
            "eltwise_jit_nCsp16c", ExecutorType::jit_x64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp16c, LayoutType::nCsp16c},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);

                VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);
                return supportsJitExecution(config);
            },
            [](const EltwiseConfig& config) -> std::optional<EltwiseConfig> {
                return requiresFallbackEltwise(config,
                                               eltwiseTypeMapping,
                                               {LayoutType::nCsp16c, LayoutType::nCsp16c},
                                               eltwiseMappingNotation);
            },
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        OV_CPU_INSTANCE_X64(
            "eltwise_jit_nCsp8c", ExecutorType::jit_x64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nCsp8c, LayoutType::nCsp8c},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);

                // return supportsJitExecution(config) && !isBitwiseAlgorithm(config);
                return supportsJitExecution(config);
            },
            RequiresFallbackDefault{{LayoutType::nCsp8c, LayoutType::nCsp8c}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        // ARM/AARCH64 JIT executor
        OV_CPU_INSTANCE_ARM64(
            "eltwise_jit_arm", ExecutorType::jit_aarch64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config) -> bool {
                return supportsJitExecution(config);
            },
            RequiredNoFallback<EltwiseAttrs>{},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        // RISC-V JIT executor
        OV_CPU_INSTANCE_RISCV64(
            "eltwise_jit_riscv", ExecutorType::jit_riscv64, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config) -> bool {
                return supportsJitExecution(config);
            },
            RequiredNoFallback<EltwiseAttrs>{},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
        )
        // ACL executor
        OV_CPU_INSTANCE_ACL(
            "eltwise_acl", ExecutorType::Acl, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config) -> bool {
                return AclEltwiseExecutor::supports(config);
            },
            RequiredNoFallback<EltwiseAttrs>{},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<AclEltwiseExecutor>(attrs, memory, context);
            }
        )
        // SHL executor
        OV_CPU_INSTANCE_SHL(
            "eltwise_shl", ExecutorType::Shl, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                return ShlEltwiseExecutor::supports(config);
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<ShlEltwiseExecutor>(attrs, memory, context);
            }
        )
        // SHL executor
        OV_CPU_INSTANCE_SHL(
            "eltwise_shl", ExecutorType::Shl, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                return ShlEltwiseExecutor::supports(config);
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<ShlEltwiseExecutor>(attrs, memory, context);
            }
        )
        // Reference executor
        OV_CPU_INSTANCE_COMMON(
            "eltwise_ref", ExecutorType::Common, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::ncsp, LayoutType::ncsp},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                return supportsRefExecution(config);
            },
            RequiresFallbackDefault{{LayoutType::ncsp, LayoutType::ncsp}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
            )
        // Reference executor
        OV_CPU_INSTANCE_COMMON(
            "eltwise_ref", ExecutorType::Common, OperationType::Eltwise, ShapeTolerance::Agnostic,
            // supports
            [](const EltwiseConfig& config, const MemoryFormatFilter& memoryFormatFilter) -> bool {
                VERIFY(MatchesMemoryFormatFilter(config.descs, LayoutConfig{LayoutType::nspc, LayoutType::nspc},
                                                 memoryFormatFilter, eltwiseMappingNotation), MEMORY_FORMAT_MISMATCH);
                return supportsRefExecution(config);
            },
            RequiresFallbackDefault{{LayoutType::nspc, LayoutType::nspc}},
            AcceptsAnyShape<EltwiseAttrs>{},
            [](const EltwiseAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) {
                return std::make_shared<EltwiseStateFulExecutor>(attrs, memory, context);
            }
            )
    };

    return eltwiseImplementations;
}

}  // namespace ov::intel_cpu
