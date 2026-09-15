// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine.hpp"

namespace cldnn {
namespace vulkan {

std::shared_ptr<cldnn::engine> create_vulkan_engine(const device::ptr& device, runtime_types runtime_type);

}  // namespace vulkan
}  // namespace cldnn
