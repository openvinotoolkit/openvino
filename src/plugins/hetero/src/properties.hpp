// Copyright (C) 2018-2026 Intel Corporation
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
 * @brief Selects the export_model() blob format: true = HSM container, false = legacy framed layout (default).
 * @note Transition switch; removed together with the legacy format. Set it on the plugin (not only per compile call)
 * so that Core caching picks up the matching blob format.
 */
static constexpr Property<bool, PropertyMutability::RW> export_hsm_format{"HETERO_EXPORT_HSM_FORMAT"};
}  // namespace hetero
}  // namespace ov
