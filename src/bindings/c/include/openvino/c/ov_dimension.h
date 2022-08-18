// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_dimension C API, which is a C wrapper for ov::Dimension class.
 *
 * @file ov_dimension.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @struct ov_dimension
 * @brief This is a structure interface equal to ov::Dimension
 */
typedef struct ov_dimension {
    int64_t min;  //!< The lower inclusive limit for the dimension.
    int64_t max;  //!< The upper inclusive limit for the dimension.
} ov_dimension_t;

// Dimension
/**
 * @defgroup dimension dimension
 * @ingroup openvino_c
 * Set of functions representing of Dimension.
 * @{
 */

/**
 * @brief Init a static dimension object
 * @ingroup dimension
 * @param dimension The input pointer of dimension.
 * @param dimension_value The dimension value for this object
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e) ov_dimension_init(ov_dimension_t* dimension, int64_t dimension_value);

/**
 * @brief Init a dynamic dimension object
 * @ingroup dimension
 * @param dimension The input pointer of dimension.
 * @param min_dimension The lower inclusive limit for the dimension, for static object you should set same value(>=0)
 * with max_dimension
 * @param max_dimension The upper inclusive limit for the dimension, for static object you should set same value(>=0)
 * with min_dimension
 * @return Status code of the operation: OK(0) for success
 */
OPENVINO_C_API(ov_status_e)
ov_dimension_init_dynamic(ov_dimension_t* dimension, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Check this dimension whether is dynamic
 * @ingroup dimension
 * @param dim The dimension pointer that will be checked.
 * @return Boolean, true is dynamic and false is static.
 */
OPENVINO_C_API(bool) ov_dimension_is_dynamic(const ov_dimension_t* dim);

/** @} */  // end of Dimension
