// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Template plugin.
 * These properties should be used in set_property() and compile_model() methods of plugins
 *
 * @file template/properties.hpp
 */

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace template_plugin {

// ! [properties:public_header]

/**
 * @brief Allows to disable all transformations for execution inside the TEMPLATE plugin.
 */
static constexpr Property<bool, PropertyMutability::RW> disable_transformations{"DISABLE_TRANSFORMATIONS"};

// ! [properties:public_header]

}  // namespace template_plugin
}  // namespace ov
