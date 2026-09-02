// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

struct SelectiveSSMAttrs {};

using SelectiveSSMConfig = executor::Config<SelectiveSSMAttrs>;

enum SelectiveSSMArgId : uint8_t {
    ARG_SSM_A = ARG_SRC_0,
    ARG_SSM_DT = ARG_SRC_1,
    ARG_SSM_B = ARG_SRC_2,
    ARG_SSM_X = ARG_SRC_3,
    ARG_SSM_C = ARG_SRC_4,
    ARG_SSM_STATE = ARG_SRC_5,
    ARG_SSM_OUT = ARG_DST_0,
    ARG_SSM_OUT_STATE = ARG_DST_0 + 1,
};

}  // namespace ov::intel_cpu
