// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in set_property() and compile_model() methods of plugins
 *
 * @file template/config.hpp
 */

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace template_plugin {

// ! [public_header:properties]

/**
 * @brief Defines the number of throutput streams used by TEMPLATE plugin.
 */
static constexpr Property<uint32_t, PropertyMutability::RW> throughput_streams{"THROUGHPUT_STREAMS"};

// ! [public_header:properties]

}  // namespace template_plugin
}  // namespace ov
