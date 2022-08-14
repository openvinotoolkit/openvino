// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for partial shape C API, which is a C wrapper for ov::PartialShape class.
 *
 * @file ov_partial_shape.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_dimension.h"
#include "openvino/c/ov_layout.h"
#include "openvino/c/ov_rank.h"
#include "openvino/c/ov_shape.h"

typedef struct ov_partial_shape ov_partial_shape_t;

// PartialShape
/**
 * @defgroup partial_shape partial_shape
 * @ingroup openvino_c
 * Set of functions representing PartialShape.
 * @{
 */

/**
 * @brief Create a partial shape and initialze with rank and dimension.
 * @ingroup partial_shape
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
 * @ingroup partial_shape
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(const char*) ov_partial_shape_to_string(ov_partial_shape_t* partial_shape);

/**
 * @brief Release partial shape.
 * @ingroup partial_shape
 * @param partial_shape will be released.
 */
OPENVINO_C_API(void) ov_partial_shape_free(ov_partial_shape_t* partial_shape);

/**
 * @brief Covert partial shape to static shape.
 * @ingroup partial_shape
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape);

/**
 * @brief Covert shape to partial shape.
 * @ingroup shape
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t** partial_shape);

/** @} */  // end of partial_shape
