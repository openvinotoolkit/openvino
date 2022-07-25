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
typedef struct ov_rank ov_rank_t;

// Rank
/**
 * @defgroup rank rank
 * @ingroup openvino_c
 * Set of functions representing of rank.
 * @{
 */

/**
 * @brief Create a rank object
 * @ingroup rank
 * @param min_dimension The lower inclusive limit for the dimension, for static object you should set same value(>=0)
 * with max_dimension
 * @param max_dimension The upper inclusive limit for the dimension, for static object you should set same value(>=0)
 * with min_dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_create(ov_rank_t** rank, int64_t min_dimension, int64_t max_dimension);

/**
 * @brief Create a static rank object
 * @ingroup rank
 * @param rank_value The rank value for this object
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e)
ov_rank_static_create(ov_rank_t** rank, int64_t rank_value);

/**
 * @brief Release rank object.
 * @ingroup rank
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_rank_free(ov_rank_t* rank);

/** @} */  // end of Rank
