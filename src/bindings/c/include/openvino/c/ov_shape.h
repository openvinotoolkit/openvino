// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for ov_shape C API
 *
 * @file ov_shape.h
 */

#pragma once

#define MAX_DIMENSION 8

/**
 * @struct ov_shape_t
 * @brief Reprents a static shape.
 */
typedef struct {
    int64_t rank;
    int64_t dims[MAX_DIMENSION];
} ov_shape_t;
