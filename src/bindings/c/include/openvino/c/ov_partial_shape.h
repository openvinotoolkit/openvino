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

/**
 * @struct ov_partial_shape
 * @brief It represents a shape that may be partially or totally dynamic.
 * A PartialShape may have:
 * Dynamic rank. (Informal notation: `?`)
 * Static rank, but dynamic dimensions on some or all axes.
 *     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
 * Static rank, and static dimensions on all axes.
 *     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
 *
 * An interface to make user can initialize ov_partial_shape_t
 */
typedef struct ov_partial_shape {
    ov_rank_t rank;
    ov_dimension_t* dims;
} ov_partial_shape_t;

// PartialShape
/**
 * @defgroup partial_shape partial_shape
 * @ingroup openvino_c
 * Set of functions representing PartialShape.
 * @{
 */

/**
 * @brief Initialze a partial shape with static rank and dynamic dimension.
 * @ingroup partial_shape
 * @param rank support static rank
 * @param dims support dynamic and static dimension
 *  Static rank, but dynamic dimensions on some or all axes.
 *     Examples: "{1,2,?,4}" or "{?,?,?}" or "{1,2,-1,4}""
 *  Static rank, and static dimensions on all axes.
 *     Examples: "{1,2,3,4}" or "{6}" or "{}""
 *
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_partial_shape_init(ov_partial_shape_t* partial_shape_obj, int64_t rank, ov_dimension_t* dims);

/**
 * @brief Initialze a partial shape with dynamic rank and dynamic dimension.
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
ov_partial_shape_init_dynamic_rank(ov_partial_shape_t* partial_shape_obj, ov_rank_t rank, ov_dimension_t* dims);

/**
 * @brief Initialize a partial shape with static rank and static dimension.
 * @ingroup partial_shape
 * @param rank support static rank
 * @param dims support static dimension
 *  Static rank, and static dimensions on all axes.
 *     Examples: "{1,2,3,4}" or "{6}" or "{}""
 *
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_partial_shape_init_static_dimension(ov_partial_shape_t* partial_shape_obj, int64_t rank, int64_t* dims);

/**
 * @brief Release partial shape internal memory.
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
 * @ingroup partial_shape
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t* partial_shape);

/**
 * @brief Help function, convert a partial shape to readable string.
 * @ingroup partial_shape
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(const char*) ov_partial_shape_to_string(const ov_partial_shape_t* partial_shape);

/** @} */  // end of partial_shape
