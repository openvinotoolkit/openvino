// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "nodes/executors/concat.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementations.hpp"

#if defined(OV_CPU_WITH_ACL)
#    include "nodes/executors/acl/acl_concat.hpp"
#    include "nodes/executors/implementation_utils.hpp"
#endif

namespace ov::intel_cpu {

template <>
const std::vector<ExecutorImplementation<ConcatAttrs>>& getImplementations() {
#if defined(OV_CPU_WITH_ACL)
    static const std::vector<ExecutorImplementation<ConcatAttrs>> concatImplementations{
        {"concat_acl_ncsp",
         ExecutorType::Acl,
         OperationType::Concat,
         [](const executor::Config<ConcatAttrs>& config,
            [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
             return AclConcatExecutor::supports(config, LayoutType::ncsp);
         },
         HasNoOptimalConfig<ConcatAttrs>{},
         AcceptsAnyShape<ConcatAttrs>,
         CreateDefault<AclConcatExecutor, ConcatAttrs>{}},
        {"concat_acl_nspc",
         ExecutorType::Acl,
         OperationType::Concat,
         [](const executor::Config<ConcatAttrs>& config,
            [[maybe_unused]] const MemoryFormatFilter& memoryFormatFilter) -> bool {
             return AclConcatExecutor::supports(config, LayoutType::nspc);
         },
         HasNoOptimalConfig<ConcatAttrs>{},
         AcceptsAnyShape<ConcatAttrs>,
         CreateDefault<AclConcatExecutor, ConcatAttrs>{}}};
#else
    static const std::vector<ExecutorImplementation<ConcatAttrs>> concatImplementations{};
#endif
    return concatImplementations;
}

}  // namespace ov::intel_cpu
