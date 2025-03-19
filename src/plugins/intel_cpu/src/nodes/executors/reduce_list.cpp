// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_list.hpp"

namespace ov::intel_cpu {

const std::vector<ReduceExecutorDesc>& getReduceExecutorsList() {
    static std::vector<ReduceExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclReduceExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
