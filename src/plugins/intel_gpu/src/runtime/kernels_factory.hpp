// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "ocl/ocl_common.hpp"

#include <memory>

namespace cldnn {

namespace kernels_factory {

// Creates instance of kernel for selected engine type.
// For ocl engine it creates a copy of kernel object
std::shared_ptr<kernel> create(engine& engine, cl_context context, cl_kernel kernel, kernel_id kernel_id);

}  // namespace kernels_factory
}  // namespace cldnn
