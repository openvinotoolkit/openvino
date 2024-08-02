// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/itt.hpp"

namespace intel_npu {
namespace itt {
namespace domains {

OV_ITT_DOMAIN(NPUPlugin);
OV_ITT_DOMAIN(LevelZeroBackend);

}  // namespace domains
}  // namespace itt
}  // namespace intel_npu

namespace itt = ::intel_npu::itt;
