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
 * @brief Check this rank whether is dynamic
 * @ingroup rank
 * @param rank The rank pointer that will be checked.
 * @return bool The return value.
 */
OPENVINO_C_API(bool) ov_rank_is_dynamic(const ov_rank_t rank);

/** @} */  // end of Rank
