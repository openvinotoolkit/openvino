// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<InterpolateExecutorDesc>& getInterpolateExecutorsList() {
    static std::vector<InterpolateExecutorDesc> descs = {
            OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<ACLInterpolateExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov