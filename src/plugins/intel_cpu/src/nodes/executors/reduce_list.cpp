// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_list.hpp"

#include <vector>

#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include <memory>

#    include "nodes/executors/acl/acl_reduce.hpp"
#    include "nodes/executors/executor.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<ReduceExecutorDesc>& getReduceExecutorsList() {
    static std::vector<ReduceExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclReduceExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
