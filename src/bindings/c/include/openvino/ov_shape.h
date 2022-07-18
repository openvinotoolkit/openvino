// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_shape.h
 */

#pragma once

#include "ov_common.h"

// Shape
/**
 * @defgroup Shape Shape
 * @ingroup openvino_c
 * Set of functions representing of Shape, PartialShape, etc.
 * @{
 */

OPENVINO_C_API(ov_status_e) ov_rank_create(ov_rank_t** rank, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Release rank object.
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_rank_free(ov_rank_t* rank);

/**
 * @brief Create a dimensions object
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_create(ov_dimensions_t** dimensions);

/**
 * @brief Release a dimensions object
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_dimensions_free(ov_dimensions_t* dimensions);

/**
 * @brief Add a dimension with bounded range into dimensions
 * @ingroup Tensor
 * @param min_dimension The lower inclusive limit for the dimension
 * @param max_dimension The upper inclusive limit for the dimension
 *
 * Static dimension: min_dimension == max_dimension > 0
 * Dynamic dimension:
 *     min_dimension == -1 ? 0 : min_dimension
 *     max_dimension == -1 ? Interval::s_max : max_dimension
 *
 */
OPENVINO_C_API(ov_status_e) ov_dimensions_add(ov_dimensions_t* dimension, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Create a partial shape and initialze with rank and dimension.
 * @ingroup Tensor
 * @param rank support dynamic and static rank
 * @param dims support dynamic and static dimension
 *  Dynamic rank:
 *     Example: "?"
 *  Static rank, but dynamic dimensions on some or all axes.
 *     Examples: "{1,2,?,4}" or "{?,?,?}" or "{1,2,-1,4}""
 *  Static rank, and static dimensions on all axes.
 *     Examples: "{1,2,3,4}" or "{6}" or "{}""
 *
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_partial_shape_create(ov_partial_shape_t** partial_shape_obj, ov_rank_t* rank, ov_dimensions_t* dims);

/**
 * @brief Parse the partial shape to readable string.
 * @ingroup Tensor
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_partial_shape_to_string(ov_partial_shape_t* partial_shape);

/**
 * @brief Release partial shape.
 * @ingroup Tensor
 * @param partial_shape will be released.
 */
OPENVINO_C_API(void) ov_partial_shape_free(ov_partial_shape_t* partial_shape);

/**
 * @brief Covert partial shape to static shape.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape);

/**
 * @brief Covert shape to partial shape.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t** partial_shape);

/**
 * @brief Create a layout object.
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_layout_create(ov_layout_t** layout, const char* layout_desc);

/**
 * @brief Free layout object.
 * @param layout will be released.
 */
OPENVINO_C_API(void) ov_layout_free(ov_layout_t* layout);

/**
 * @brief Convert layout object to a readable string.
 * @param layout will be converted.
 * @return string that describes the layout content.
 */
OPENVINO_C_API(const char*) ov_layout_to_string(ov_layout_t* layout);

/** @} */  // end of Shape
