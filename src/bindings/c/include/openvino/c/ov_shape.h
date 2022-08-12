// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_shape.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @struct ov_shape_t
 * @brief Reprents a static shape.
 */
typedef struct {
    int64_t rank;
    int64_t* dims;
} ov_shape_t;

/**
 * @brief Init a shape object, allocate space for its dimensions.
 * @ingroup shape
 * @param rank The rank value for this object, it should be more than 0(>0)
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_shape_init(ov_shape_t* shape, int64_t rank);

/**
 * @brief Free a shape object's internal memory
 * @ingroup shape
 * @param ov_status_e a status code.
 */
OPENVINO_C_API(ov_status_e) ov_shape_deinit(ov_shape_t* shape);