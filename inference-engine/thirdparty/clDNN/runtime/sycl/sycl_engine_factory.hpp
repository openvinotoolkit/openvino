// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/device.hpp"
#include "cldnn/runtime/engine.hpp"

#include <memory>

namespace cldnn {
namespace sycl {

std::shared_ptr<cldnn::engine> create_sycl_engine(const device::ptr device, runtime_types runtime_type, const engine_configuration& configuration);

}  // namespace sycl
}  // namespace cldnn
