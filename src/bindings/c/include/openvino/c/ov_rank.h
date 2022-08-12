// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_rank.h
 */

#pragma once

#include "openvino/c/ov_common.h"
typedef struct ov_rank ov_rank_t;

// Rank
/**
 * @defgroup rank rank
 * @ingroup openvino_c
 * Set of functions representing of rank.
 * @{
 */

/**
 * @brief Create a static rank object
 * @ingroup rank
 * @param rank_value The rank value for this object, it should be not less than 0(>=0)
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_create(ov_rank_t** rank, int64_t rank_value);

/**
 * @brief Create a dynamic rank object
 * @ingroup rank
 * @param min_rank The lower inclusive limit for the rank
 * @param max_rank The upper inclusive limit for the rank
 * with min_dimension
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_create_dynamic(ov_rank_t** rank, int64_t min_rank, int64_t max_rank);

/**
 * @brief Release rank object.
 * @ingroup rank
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(void) ov_rank_free(ov_rank_t* rank);

/** @} */  // end of Rank
