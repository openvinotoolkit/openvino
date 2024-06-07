// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "support/Mutex.h"

void ov::intel_cpu::configureThreadSafe(const std::function<void(void)>& config) {
    // Issue: CVS-123514
    static arm_compute::Mutex mtx_config;
    arm_compute::lock_guard<arm_compute::Mutex> _lock{mtx_config};
    config();
}
