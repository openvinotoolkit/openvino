// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace hetero {
/**
 * @brief Read-only property to get device caching properties
 */
static constexpr Property<std::string, PropertyMutability::RO> caching_device_properties{"CACHING_DEVICE_PROPERTIES"};

}  // namespace hetero
}  // namespace ov
