// Copyright (C) 2018-2024 Intel Corporation
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

/**
 * @brief Read-only property showing number of compiled submodels
 */
static constexpr Property<size_t, PropertyMutability::RO> number_of_submodels{"HETERO_NUMBER_OF_SUBMODELS"};
}  // namespace hetero
}  // namespace ov
