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
#include "ov_optional.hpp"
#include "utils/cpp/maybe_unused.hpp"
#include "utils/debug_capabilities.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#include "nodes/executors/x64/jit_mvn.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_mvn.hpp"
#endif

#include "nodes/executors/common/ref_mvn.hpp"

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

using LayoutConfig = std::vector<LayoutType>;

// clang-format off
static const TypeMapping aclMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_f32 | _f16, _any}, pt(bypass(), use<0>())},
        {{_any, _any},        pt(just<f32>(), just<f32>())}
};

static const TypeMapping jitMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_f32 | _f16, _any}, pt(bypass(), use<0>())},
        {{_any, _any},        pt(just<f32>(), just<f32>())}
};

static const TypeMapping refMVNTypeMapping {
        // {src, dst}         pt<src, dst>
        {{_any, _any},        pt(just<f32>(), just<f32>())}
};

static const MappingNotation mvnMappingNotation {ARG_SRC, ARG_DST};
// clang-format on

static bool fullyMatchConfiguration(const MemoryDescArgs& currentDescriptors,
                                    const InOutTypes& typeConfig,
                                    const LayoutConfig& layoutConfig,
                                    const MappingNotation& notation) {
    for (size_t i = 0; i < typeConfig.size(); i++) {
        const auto& type = typeConfig[i];
        const auto& desc = currentDescriptors.at(notation[i]);

        if (desc->empty())
            continue;

        if (desc->getPrecision() != type)
            return false; // type mismatch

        if (!desc->hasLayoutType(layoutConfig[i]))
            return false; // layout mismatch
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

        if (desc->empty())
            continue;

        if (descType == type && desc->hasLayoutType(layout)) {
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

    return ov::optional<executor::Config<Attrs>>(MVNConfig {optimalDescriptors, config.attrs, config.postOps});
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noPostOps(const MVNConfig& config) {
    return config.postOps.empty();
}

template <>
const std::vector<ExecutorImplementation<MVNAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<MVNAttrs>> mvnImplementations {
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_ncsp",
            ExecutorType::jit_x64,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, postOps, memory, context);
        })
        OV_CPU_INSTANCE_X64(
            "mvn_jit_x64_nspc",
            ExecutorType::jit_x64,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                return JITMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              jitMVNTypeMapping,
                                              {LayoutType::nspc, LayoutType::nspc},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<JITMVNExecutor>(attrs, postOps, memory, context);
        })
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_nspc",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                if (!config.descs.at(ARG_SRC)->hasLayoutType(LayoutType::nspc)) return false;
                return ACLMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              aclMVNTypeMapping,
                                              {LayoutType::nspc, LayoutType::nspc},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<ACLMVNExecutor>(attrs, postOps, memory, context);
        })
        OV_CPU_INSTANCE_ACL(
            "mvn_acl_ncsp",
            ExecutorType::Acl,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                if (!config.descs.at(ARG_SRC)->hasLayoutType(LayoutType::ncsp)) return false;
                return ACLMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              aclMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<ACLMVNExecutor>(attrs, postOps, memory, context);
        })
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref_ncsp",
            ExecutorType::Common,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                if (!config.descs.at(ARG_SRC)->hasLayoutType(LayoutType::ncsp)) return false;
                return CommonMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              refMVNTypeMapping,
                                              {LayoutType::ncsp, LayoutType::ncsp},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<CommonMVNExecutor>(attrs, postOps, memory, context);
        })
        OV_CPU_INSTANCE_COMMON(
            "mvn_ref_nspc",
            ExecutorType::Common,
            OperationType::MVN,
            ShapeTolerance::Agnostic,
            // supports
            [](const MVNConfig& config) -> bool {
                if (!config.descs.at(ARG_SRC)->hasLayoutType(LayoutType::nspc)) return false;
                return CommonMVNExecutor::supports(config);
            },
            // requiresFallback
            [](const MVNConfig& config) -> ov::optional<executor::Config<MVNAttrs>> {
                return requiresFallbackCommon(config,
                                              refMVNTypeMapping,
                                              {LayoutType::nspc, LayoutType::nspc},
                                              mvnMappingNotation);
            },
            // acceptsShapes
            [](const MemoryArgs& memory) -> bool {
                // @todo create syntactic sugar (functor) for shape agnostic lambda
                return true;
            },
            // create
            [](const MVNAttrs& attrs,
               const PostOps& postOps,
               const MemoryArgs& memory,
               const ExecutorContext::CPtr context) {
                return std::make_shared<CommonMVNExecutor>(attrs, postOps, memory, context);
        })
    };
    return mvnImplementations;
}
}  // namespace intel_cpu
}  // namespace ov
