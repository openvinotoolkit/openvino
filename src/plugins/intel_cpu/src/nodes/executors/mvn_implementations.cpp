// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <optional>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "mvn_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/arch_macros.h"
#include "utils/debug_capabilities.h"

#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/executors/x64/jit_mvn.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_mvn.hpp"
#endif

#include "nodes/executors/common/ref_mvn.hpp"

namespace ov::intel_cpu {

using namespace TypeMaskAlias;
using namespace executor;

// Mapping notation for MVN arguments
static const MappingNotation mvnMappingNotation{{ARG_SRC, 0}, {ARG_DST, 1}};

// Layout configuration for MVN - support planar, channel-last and blocked formats
using LayoutConfig = std::vector<LayoutType>;
static const LayoutConfig mvnPlanarLayoutConfig{LayoutType::ncsp, LayoutType::ncsp};
static const LayoutConfig mvnByChannelLayoutConfig{LayoutType::nspc, LayoutType::nspc};
static const LayoutConfig mvnBlockedC8LayoutConfig{LayoutType::nCsp8c, LayoutType::nCsp8c};
static const LayoutConfig mvnBlockedC16LayoutConfig{LayoutType::nCsp16c, LayoutType::nCsp16c};

// Type mapping for MVN - supports f32, bf16, f16, i8, u8
static const TypeMapping mvnTypeMapping{
    // {src, dst}                                   pt<src, dst>
    {{_f32, _f32}, {bypass(), bypass()}},
    {{_bf16, _bf16}, {bypass(), bypass()}},
    {{_f16, _f16}, {bypass(), bypass()}},
    {{_i8, _f32}, {bypass(), bypass()}},  // i8 input -> f32 output
    {{_u8, _f32}, {bypass(), bypass()}},  // u8 input -> f32 output
    {{_i8, _i8}, {bypass(), bypass()}},   // i8 input -> i8 output
    {{_u8, _u8}, {bypass(), bypass()}},   // u8 input -> u8 output
    // Special handling for bf16 with blocked layouts to prevent precision loss
    {{_bf16, _f32}, {bypass(), just<ov::element::f32>()}},  // bf16 -> f32 conversion
    {{_f32, _bf16}, {just<ov::element::f32>(), bypass()}},  // f32 -> bf16 conversion
    // Fallback to f32 for any unsupported type configuration
    {{_any, _any}, {just<ov::element::f32>(), just<ov::element::f32>()}},
};

/**
 * @brief MVN Executor Implementation Registry
 *
 * This file defines available MVN executor implementations:
 *
 * 1. JIT x64 Executor (Intel x86_64 only):
 *    - Optimized implementation using JIT compilation
 *    - Supports all precisions (f32, bf16, f16, i8, u8)
 *    - Supports all layouts (planar, blocked, channel-last)
 *    - Best performance on Intel CPUs with AVX2/AVX512
 *
 * 2. ACL Executor (ARM only):
 *    - Uses ARM Compute Library (NEMeanStdDevNormalizationLayer)
 *    - Supports f32 and f16 precisions only
 *    - Requires normalizeVariance=true and INSIDE_SQRT mode
 *    - Optimized for ARM NEON/SVE architectures
 *
 * 3. Reference Executor (fallback):
 *    - Generic C++ implementation
 *    - Supports all configurations
 *    - Used when specialized executors are not available
 *
 * Selection priority: JIT (if x64) -> ACL (if ARM) -> Reference
 */

// to keep OV_CPU_INSTANCE macros aligned
// clang-format off
template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MVNAttrs>> mvnImplementations {
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_ncsp",
            ExecutorType::Jit,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have planar layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::ncsp) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::ncsp)) {
                    return false;
                }
                bool supported = MVNJitExecutor::supports(config);
                DEBUG_LOG("MVN JIT x64 ncsp executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnPlanarLayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN JIT x64 ncsp executor");
                return std::make_shared<MVNJitExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nspc",
            ExecutorType::Jit,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have channel-last layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::nspc) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::nspc)) {
                    return false;
                }
                bool supported = MVNJitExecutor::supports(config);
                DEBUG_LOG("MVN JIT x64 nspc executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnByChannelLayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN JIT x64 nspc executor");
                return std::make_shared<MVNJitExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nCsp8c",
            ExecutorType::Jit,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have blocked 8c layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::nCsp8c) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::nCsp8c)) {
                    return false;
                }
                bool supported = MVNJitExecutor::supports(config);
                DEBUG_LOG("MVN JIT x64 nCsp8c executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnBlockedC8LayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN JIT x64 nCsp8c executor");
                return std::make_shared<MVNJitExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nCsp16c",
            ExecutorType::Jit,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have blocked 16c layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::nCsp16c) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::nCsp16c)) {
                    return false;
                }
                bool supported = MVNJitExecutor::supports(config);
                DEBUG_LOG("MVN JIT x64 nCsp16c executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnBlockedC16LayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN JIT x64 nCsp16c executor");
                return std::make_shared<MVNJitExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_ncsp",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have planar layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::ncsp) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::ncsp)) {
                    return false;
                }
                bool supported = ACLMVNExecutor::supports(config);
                DEBUG_LOG("MVN ACL ncsp executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnPlanarLayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN ACL ncsp executor");
                return std::make_shared<ACLMVNExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_nspc",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have channel-last layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::nspc) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::nspc)) {
                    return false;
                }
                bool supported = ACLMVNExecutor::supports(config);
                DEBUG_LOG("MVN ACL nspc executor support check: ", supported ? "supported" : "not supported");
                return supported;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnByChannelLayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN ACL nspc executor");
                return std::make_shared<ACLMVNExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref_nspc",
            ExecutorType::Common,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const executor::Config<MVNAttrs>& config) -> bool {
                // Check if both src and dst have channel-last layout
                if (!config.descs.at(ARG_SRC_0)->hasLayoutType(LayoutType::nspc) ||
                    !config.descs.at(ARG_DST)->hasLayoutType(LayoutType::nspc)) {
                    return false;
                }
                DEBUG_LOG("MVN Reference nspc executor support check: supported");
                return true;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 mvnByChannelLayoutConfig,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN Reference nspc executor");
                return std::make_shared<MVNRefExecutor>(attrs, memory, context);
            }
            )
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref",
            ExecutorType::Common,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports - always returns true as fallback
            [](const executor::Config<MVNAttrs>& /*config*/) -> bool {
                DEBUG_LOG("MVN Reference executor support check: always supported (fallback)");
                return true;
            },
            // createOptimalConfig
            [](const executor::Config<MVNAttrs>& config) -> std::optional<executor::Config<MVNAttrs>> {
                // Reference implementation accepts whatever layout is provided
                // Determine actual layouts from descriptors
                std::vector<LayoutType> actualLayouts;
                auto srcDesc = config.descs.at(ARG_SRC_0);
                auto dstDesc = config.descs.at(ARG_DST);
                
                // Default to planar if layout cannot be determined
                LayoutType srcLayout = LayoutType::ncsp;
                LayoutType dstLayout = LayoutType::ncsp;
                
                if (srcDesc->hasLayoutType(LayoutType::nspc)) {
                    srcLayout = LayoutType::nspc;
                } else if (srcDesc->hasLayoutType(LayoutType::nCsp8c)) {
                    srcLayout = LayoutType::nCsp8c;
                } else if (srcDesc->hasLayoutType(LayoutType::nCsp16c)) {
                    srcLayout = LayoutType::nCsp16c;
                }
                
                if (dstDesc->hasLayoutType(LayoutType::nspc)) {
                    dstLayout = LayoutType::nspc;
                } else if (dstDesc->hasLayoutType(LayoutType::nCsp8c)) {
                    dstLayout = LayoutType::nCsp8c;
                } else if (dstDesc->hasLayoutType(LayoutType::nCsp16c)) {
                    dstLayout = LayoutType::nCsp16c;
                }
                
                actualLayouts = {srcLayout, dstLayout};
                
                return createOptimalConfigCommon(config,
                                                 mvnTypeMapping,
                                                 actualLayouts,
                                                 mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            [](const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> std::shared_ptr<Executor> {
                DEBUG_LOG("Creating MVN Reference executor");
                return std::make_shared<MVNRefExecutor>(attrs, memory, context);
            }
            )
    };
    
    return mvnImplementations;
}
// clang-format on

}  // namespace ov::intel_cpu