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
#include "openvino/c/ov_dimension.h"

typedef ov_dimension_t ov_rank_t;

// Rank
/**
 * @defgroup rank rank
 * @ingroup openvino_c
 * Set of functions representing of rank.
 * @{
 */

/**
 * @brief Initialize a static rank object
 * @ingroup rank
 * @param rank The input rank pointer.
 * @param rank_value The rank value for this object, it should be not less than 0(>=0).
 * @return ov_status_e The return status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_init(ov_rank_t* rank, int64_t rank_value);

/**
 * @brief Initialize a dynamic rank object.
 * @ingroup rank
 * @param rank The input rank pointer.
 * @param min_rank The lower inclusive limit for the rank.
 * @param max_rank The upper inclusive limit for the rank.
 * @return ov_status_e The return status code.
 */
OPENVINO_C_API(ov_status_e) ov_rank_init_dynamic(ov_rank_t* rank, int64_t min_rank, int64_t max_rank);

/**
 * @brief Check this rank whether is dynamic
 * @ingroup rank
 * @param rank The rank pointer that will be checked.
 * @return bool The return value.
 */
OPENVINO_C_API(bool) ov_rank_is_dynamic(const ov_rank_t* rank);

/** @} */  // end of Rank
