// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/al/config/npuw.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerNPUWOptions(OptionsDesc& desc) {
    desc.add<FROM_NPUW>();
}
