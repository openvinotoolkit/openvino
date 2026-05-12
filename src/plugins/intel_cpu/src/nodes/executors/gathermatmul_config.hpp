// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "executor_config.hpp"

namespace ov::intel_cpu {

struct GatherMatmulAttrs {};

using GatherMatmulConfig = executor::Config<GatherMatmulAttrs>;

}  // namespace ov::intel_cpu
