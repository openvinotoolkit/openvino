// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/utils/enums.hpp"

namespace vpu {

// Must be synchronized with firmware side.
VPU_DECLARE_ENUM(PowerConfig,
    FULL         = 0,
    INFER        = 1,
    STAGE        = 2,
    STAGE_SHAVES = 3,
    STAGE_NCES   = 4,
)

}  // namespace vpu
