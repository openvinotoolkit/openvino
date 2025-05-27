// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "nodes/executors/type_mask.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#include "nodes/executors/x64/jit_mvn.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_mvn.hpp"
#endif

#include "nodes/executors/common/ref_mvn.hpp"

namespace ov::intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// clang-format off
static const TypeMapping aclMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_f32 | _f16 | _bf16, _any}, pt(bypass(),    use<0>())},
        {{_any, _any},                pt(just<f32>(), just<f32>())}
};

static const TypeMapping jitMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_f32 | _f16 | _bf16, _f32 | _f16 | _bf16}, pt(bypass(), use<0>())},
        {{_u8  | _i8,  _any},                        pt(bypass(), bypass())},
        {{_any, _u8  | _i8},                         pt(use<1>(), bypass())},
        {{_any, _any},                               pt(just<f32>(), use<0>())}
};

static const TypeMapping refMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_any, _any},        pt(just<f32>(), just<f32>())}
};

static const MappingNotation mvnMappingNotation {ARG_SRC, ARG_DST};
// clang-format on

template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MVNAttrs>> mvnImplementations {
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nspc", ExecutorType::jit_x64, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
//                VERIFY(one_of(srcRank(config), 4lu, 5lu), UNSUPPORTED_SRC_RANK);
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::nspc, LayoutType::nspc},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nCsp16c", ExecutorType::jit_x64, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
//                VERIFY(one_of(srcRank(config), 4lu, 5lu), UNSUPPORTED_SRC_RANK);
                VERIFY(mayiuse(cpu::x64::avx512_core), UNSUPPORTED_ISA);
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::nCsp16c, LayoutType::nCsp16c},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nCsp8c", ExecutorType::jit_x64, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
//                VERIFY(one_of(srcRank(config), 4lu, 5lu), UNSUPPORTED_SRC_RANK);
                VERIFY(mayiuse(cpu::x64::avx2), UNSUPPORTED_ISA);
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::nCsp8c, LayoutType::nCsp8c},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_ncsp", ExecutorType::jit_x64, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_nspc", ExecutorType::Acl, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return ACLMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              aclMVNTypeMapping,
                                              {LayoutType::nspc, LayoutType::nspc},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<ACLMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_ncsp", ExecutorType::Acl, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return ACLMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              aclMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<ACLMVNExecutor>(attrs, memory, context);
            })
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref_ncsp", ExecutorType::Common, OperationType::MVN, ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return CommonMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> std::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              refMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            AcceptsAnyShape<MVNAttrs>{},
            // create
            [](const MVNAttrs& attrs,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<CommonMVNExecutor>(attrs, memory, context);
            })
    };
    return mvnImplementations;
}
} // namespace ov::intel_cpu

