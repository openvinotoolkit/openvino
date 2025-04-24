// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

/**
 * @file transformations_visibility.hpp
 * @brief Defines visibility settings for OpenVINO Transformations library
 */

/**
 * @defgroup ie_transformation_api OpenVINO Transformation API
 * @brief Defines OpenVINO Transformations API which is used to transform ov::Model
 *
 * @{
 * @defgroup ie_runtime_attr_api Runtime information
 * @brief A mechanism of runtime information extension
 *
 * @defgroup ov_transformation_common_api Common optimization passes
 * @brief A set of common optimization passes
 *
 * @defgroup ie_transformation_to_opset2_api Conversion from opset3 to opset2
 * @brief A set of conversion downgrade passes from opset3 to opset2
 *
 * @defgroup ie_transformation_to_opset1_api Conversion from opset2 to opset1
 * @brief A set of conversion downgrade passes from opset2 to opset1
 * @}
 */

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY
