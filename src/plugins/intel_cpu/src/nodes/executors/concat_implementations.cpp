// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "memory_format_filter.hpp"
#include "nodes/executors/concat.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_concat.hpp"
#endif

namespace ov::intel_cpu {

template <>
const std::vector<ExecutorImplementation<ConcatAttrs>>& getImplementations() {
    static const std::vector<ExecutorImplementation<ConcatAttrs>> concatImplementations{
        OV_CPU_INSTANCE_ACL(
            "concat_acl_ncsp",
            ExecutorType::Acl,
            OperationType::Concat,
            [](const executor::Config<ConcatAttrs>& config,
               [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
                return AclConcatExecutor::supports(config, LayoutType::ncsp);
            },
            HasNoOptimalConfig<ConcatAttrs>{},
            AcceptsAnyShape<ConcatAttrs>,
            CreateDefault<AclConcatExecutor, ConcatAttrs>{})
            OV_CPU_INSTANCE_ACL(
                "concat_acl_nspc",
                ExecutorType::Acl,
                OperationType::Concat,
                [](const executor::Config<ConcatAttrs>& config,
                   [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
                    return AclConcatExecutor::supports(config, LayoutType::nspc);
                },
                HasNoOptimalConfig<ConcatAttrs>{},
                AcceptsAnyShape<ConcatAttrs>,
                CreateDefault<AclConcatExecutor, ConcatAttrs>{})
                OV_CPU_INSTANCE_COMMON(
                    "concat_ref_ncsp",
                    ExecutorType::Reference,
                    OperationType::Concat,
                    []([[maybe_unused]] const executor::Config<ConcatAttrs>& config,
                       [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
                        return false;
                    },
                    HasNoOptimalConfig<ConcatAttrs>{},
                    AcceptsAnyShape<ConcatAttrs>,
                    []([[maybe_unused]] const ConcatAttrs& attrs,
                       [[maybe_unused]] const MemoryArgs& memory,
                       [[maybe_unused]] const ExecutorContext::CPtr& context) -> ExecutorPtr {
                        return nullptr;
                    })
                    OV_CPU_INSTANCE_COMMON(
                        "concat_ref_nspc",
                        ExecutorType::Reference,
                        OperationType::Concat,
                        []([[maybe_unused]] const executor::Config<ConcatAttrs>& config,
                           [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
                            return false;
                        },
                        HasNoOptimalConfig<ConcatAttrs>{},
                        AcceptsAnyShape<ConcatAttrs>,
                        []([[maybe_unused]] const ConcatAttrs& attrs,
                           [[maybe_unused]] const MemoryArgs& memory,
                           [[maybe_unused]] const ExecutorContext::CPtr& context) -> ExecutorPtr {
                            return nullptr;
                        })};
    return concatImplementations;
}

}  // namespace ov::intel_cpu
