// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace sycl {

// Factory for sycl_engine creation.
std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type);

}  // namespace sycl
}  // namespace cldnn
