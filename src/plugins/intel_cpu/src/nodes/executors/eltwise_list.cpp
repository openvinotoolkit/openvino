// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<EltwiseExecutorDesc>& getEltwiseExecutorsList() {
    static std::vector<EltwiseExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclEltwiseExecutorBuilder>())
        OV_CPU_INSTANCE_SHL(ExecutorType::Shl, std::make_shared<ShlEltwiseExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov
