// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv_list.hpp"

namespace ov::intel_cpu {

const std::vector<DeconvExecutorDesc>& getDeconvExecutorsList() {
    static std::vector<DeconvExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclDeconvExecutorBuilder>())};

    return descs;
}

}  // namespace ov::intel_cpu
