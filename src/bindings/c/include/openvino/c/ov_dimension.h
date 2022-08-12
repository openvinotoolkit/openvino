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

typedef struct ov_dimension ov_dimension_t;
typedef struct ov_dimensions ov_dimensions_t;

// Dimension
/**
 * @defgroup dimension dimension
 * @ingroup openvino_c
 * Set of functions representing of Dimension.
 * @{
 */

/**
 * @brief Create a static dimension object
 * @ingroup dimension
 * @param dimension_value The dimension value for this object
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_dimension_create(ov_dimension_t** dimension, int64_t dimension_value);

/**
 * @brief Create a dynamic dimension object
 * @ingroup dimension
 * @param min_dimension The lower inclusive limit for the dimension, for static object you should set same value(>=0)
 * with max_dimension
 * @param max_dimension The upper inclusive limit for the dimension, for static object you should set same value(>=0)
 * with min_dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_dimension_create_dynamic(ov_dimension_t** dimension, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Release dimension object.
 * @ingroup dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_dimension_free(ov_dimension_t* dimension);

/**
 * @brief Create a dimension vector object without any items in it
 * @ingroup dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_create(ov_dimensions_t** dimensions);

/**
 * @brief Release a dimension vector object
 * @ingroup dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_dimensions_free(ov_dimensions_t* dimensions);

/**
 * @brief Add a static dimension into dimensions
 * @ingroup dimension
 * @param value The value for the dimension, it should be not less than 0(>=0)
 *
 * Static dimension: min_dimension == max_dimension >= 0
 * Dynamic dimension:
 *     min_dimension == -1 ? 0 : min_dimension
 *     max_dimension == -1 ? Interval::s_max : max_dimension
 *
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_add(ov_dimensions_t* dimension, int64_t value);

/**
 * @brief Add a dynamic dimension with bounded range into dimensions
 * @ingroup dimension
 * @param min_dimension The lower inclusive limit for the dimension, for static object you should set same value(>=0)
 * with max_dimension
 * @param max_dimension The upper inclusive limit for the dimension, for static object you should set same value(>=0)
 * with min_dimension
 *
 * Dynamic dimension:
 *     min_dimension == -1 ? 0 : min_dimension
 *     max_dimension == -1 ? Interval::s_max : max_dimension
 *
 */
OPENVINO_C_API(ov_status_e)
ov_dimensions_add_dynamic(ov_dimensions_t* dimension, int64_t min_dimension, int64_t max_dimension);

/** @} */  // end of Dimension
