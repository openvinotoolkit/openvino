// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

struct PagedSelectiveSSMAttrs {};

using PagedSelectiveSSMConfig = executor::Config<PagedSelectiveSSMAttrs>;

enum PagedSelectiveSSMArgId : uint8_t {
    ARG_PAGED_SSM_A = ARG_SRC_0,
    ARG_PAGED_SSM_DT = ARG_SRC_1,
    ARG_PAGED_SSM_B = ARG_SRC_2,
    ARG_PAGED_SSM_X = ARG_SRC_3,
    ARG_PAGED_SSM_C = ARG_SRC_4,
    ARG_PAGED_SSM_STATE = ARG_SRC_5,
    ARG_PAGED_SSM_SUBSEQUENCE_BEGINS = ARG_SRC_6,
    ARG_PAGED_SSM_BLOCK_INDICES = ARG_SRC_7,
    ARG_PAGED_SSM_BLOCK_INDICES_BEGINS = ARG_SRC_8,
    ARG_PAGED_SSM_NUM_PROCESSED_TOKENS = ARG_SRC_9,
    ARG_PAGED_SSM_CACHE_INTERVAL = ARG_SRC_10,
    ARG_PAGED_SSM_OUT = ARG_DST_0,
};

}  // namespace ov::intel_cpu
