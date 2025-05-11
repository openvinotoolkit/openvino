// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace ocl {

// Factory for ocl_engine creation. It's moved outside of ocl_engine class to avoid possible CL includes conflict
// between different engines in engine.cpp file
std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type);
#ifdef OV_GPU_WITH_SYCL
std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type);
#endif  // OV_GPU_WITH_SYCL

}  // namespace ocl
}  // namespace cldnn
