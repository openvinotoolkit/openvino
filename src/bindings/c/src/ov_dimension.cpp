// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_dimension.h"

#include "common.h"

ov_status_e ov_dimension_init_dynamic(ov_dimension_t* dim, int64_t min_dimension, int64_t max_dimension) {
    if (!dim || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_C_PARAM;
    }

    dim->max = max_dimension;
    dim->min = min_dimension;
    return ov_status_e::OK;
}

ov_status_e ov_dimension_init(ov_dimension_t* dim, int64_t dimension_value) {
    if (!dim || dimension_value <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    return ov_dimension_init_dynamic(dim, dimension_value, dimension_value);
}

bool ov_dimension_is_dynamic(const ov_dimension_t* dim) {
    if (dim->min == dim->max && dim->max > 0)
        return false;
    return true;
}