// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/concat.hpp"
#include "nodes/executors/executor_config.hpp"

namespace ov::intel_cpu {

using ConcatConfig = executor::Config<ConcatAttrs>;

}  // namespace ov::intel_cpu
