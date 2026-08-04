// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"

namespace cldnn {
namespace vk {

// Factory for vk_engine creation.
std::shared_ptr<cldnn::engine> create_vk_engine(const device::ptr device, runtime_types runtime_type);

}  // namespace vk
}  // namespace cldnn
