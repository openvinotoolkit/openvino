// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/visibility.hpp"

/**
 * @file transformations_visibility.hpp
 * @brief Defines visibility settings for Inference Engine Transformations library
 */

/**
 * @defgroup ie_transformation_api Inference Engine Transformation API
 * @brief Defines Inference Engine Transformations API which is used to transform ngraph::Function
 *
 * @{
 * @defgroup ie_runtime_attr_api Runtime information
 * @brief A mechanism of runtime information extension
 *
 * @defgroup ie_transformation_common_api Common optimization passes
 * @brief A set of common optimization passes
 *
 * @defgroup ie_transformation_to_opset2_api Conversion from opset3 to opset2
 * @brief A set of conversion downgrade passes from opset3 to opset2
 *
 * @defgroup ie_transformation_to_opset1_api Conversion from opset2 to opset1
 * @brief A set of conversion downgrade passes from opset2 to opset1
 * @}
 */

#ifdef inference_engine_transformations_EXPORTS
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_EXPORT
#else
#define TRANSFORMATIONS_API NGRAPH_HELPER_DLL_IMPORT
#endif
