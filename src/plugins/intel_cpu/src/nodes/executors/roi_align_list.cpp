// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_align_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<ROIAlignExecutorDesc>& getROIAlignExecutorsList() {
    static std::vector<ROIAlignExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclROIAlignExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov