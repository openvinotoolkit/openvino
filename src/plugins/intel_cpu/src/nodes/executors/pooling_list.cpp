// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_list.hpp"

#include <vector>

#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include <memory>

#    include "nodes/executors/acl/acl_pooling.hpp"
#    include "nodes/executors/executor.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<PoolingExecutorDesc>& getPoolingExecutorsList() {
    static std::vector<PoolingExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclPoolingExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
