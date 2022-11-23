// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_plugin_config.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @brief Read-only property to get GPU driver version
 */
static constexpr Property<std::string, PropertyMutability::RO> driver_version{"GPU_DRIVER_VERSION"};

}  // namespace intel_gpu
}  // namespace ov
