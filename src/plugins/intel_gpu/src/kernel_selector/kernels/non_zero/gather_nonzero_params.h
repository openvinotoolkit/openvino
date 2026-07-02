// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct gather_nonzero_params : public base_params {
    gather_nonzero_params() : base_params(KernelType::GATHER_NONZERO) {}
    int32_t ov_input_rank = -1;
};
}  // namespace kernel_selector
