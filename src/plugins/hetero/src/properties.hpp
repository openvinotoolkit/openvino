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

/**
 * @brief Read-only property showing number of compiled submodels
 */
static constexpr Property<size_t, PropertyMutability::RO> number_of_submodels{"HETERO_NUMBER_OF_SUBMODELS"};

/**
 * @brief Read-write property to enable/disable HETERO query model by device type and memory
 */
static constexpr Property<bool, PropertyMutability::RW> hetero_query_model_by_device{"HETERO_QUERY_MODEL_BY_DEVICE"};
}  // namespace hetero
}  // namespace ov
