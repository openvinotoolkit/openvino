// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/device.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/stream.hpp"

namespace cldnn {
namespace ocl {

// Factory for ocl_engine creation. It's moved outside of ocl_engine class to avoid possible CL includes conflict
// between different engines in engine.cpp file
std::shared_ptr<cldnn::engine> create_ocl_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration);

}  // namespace ocl
}  // namespace cldnn
