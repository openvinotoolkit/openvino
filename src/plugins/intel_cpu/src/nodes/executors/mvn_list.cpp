// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_list.hpp"

#include <vector>

#include "utils/arch_macros.h"

#if defined(OV_CPU_WITH_ACL)
#    include <memory>

#    include "nodes/executors/acl/acl_mvn.hpp"
#    include "nodes/executors/executor.hpp"
#endif

namespace ov::intel_cpu {

const std::vector<MVNExecutorDesc>& getMVNExecutorsList() {
    static std::vector<MVNExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclMVNExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
