// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Template plugin.
 * These properties should be used in set_property() and compile_model() methods of Core object
 *
 * @file template_config.hpp
 */

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {

namespace template_plugin {

// ! [public_header:properties]
/**
 * @brief Defines whether current Template device instance supports hardware blocks for fast convolution computations.
 */
static constexpr Property<bool, PropertyMutability::RO> hardware_convolugion{"HARDWARE_CONVOLUTION"};
/**
 * @brief Defines the number of throutput streams used by TEMPLATE plugin.
 */
static constexpr Property<uint32_t> throughput_streams{"THROUGHPUT_STREAMS"};
// ! [public_header:properties]

}  // namespace template_plugin
}  // namespace ov
