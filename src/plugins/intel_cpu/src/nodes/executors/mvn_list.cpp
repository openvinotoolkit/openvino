// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_list.hpp"

namespace ov::intel_cpu {

const std::vector<MVNExecutorDesc>& getMVNExecutorsList() {
    static std::vector<MVNExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclMVNExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
