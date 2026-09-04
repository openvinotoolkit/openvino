// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "kernel_selector/kernels/eltwise/eltwise_kernel_base.h"

namespace cldnn {

kernel_selector::EltwiseMode convert_to_eltwise_mode(eltwise_mode mode);
kernel_selector::eltwise_params lower_eltwise_params(const kernel_impl_params& impl_params,
                                                     kernel_selector::eltwise_params params);
kernel_impl_params canonicalize_eltwise_shapes(const kernel_impl_params& impl_params);

}  // namespace cldnn
