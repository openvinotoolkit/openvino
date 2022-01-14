// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu/utils/enums.hpp"

namespace vpu {

VPU_DECLARE_ENUM(MovidiusDdrType,
    AUTO        = 0,
    MICRON_2GB  = 1,
    SAMSUNG_2GB = 2,
    HYNIX_2GB   = 3,
    MICRON_1GB  = 4,
)

}  // namespace vpu
