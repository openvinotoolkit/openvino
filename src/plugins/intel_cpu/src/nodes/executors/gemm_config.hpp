// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "gemm_attrs.hpp"

namespace ov {
namespace intel_cpu {
using GEMMConfig = ov::intel_cpu::executor::Config<GEMMAttrs>;
}  // namespace intel_cpu
}  // namespace ov
