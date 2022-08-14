// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_dimension.h"

#include "common.h"

ov_status_e ov_dimension_create_dynamic(ov_dimension_t** dim, int64_t min_dimension, int64_t max_dimension) {
    if (!dim || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_dimension_t> _dim(new ov_dimension_t);
        if (min_dimension != max_dimension) {
            _dim->object = ov::Dimension(min_dimension, max_dimension);
        } else {
            if (min_dimension > -1) {
                _dim->object = ov::Dimension(min_dimension);
            } else {
                _dim->object = ov::Dimension();
            }
        }
        *dim = _dim.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_dimension_create(ov_dimension_t** dim, int64_t dimension_value) {
    if (!dim || dimension_value <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    return ov_dimension_create_dynamic(dim, dimension_value, dimension_value);
}

void ov_dimension_free(ov_dimension_t* dim) {
    if (dim)
        delete dim;
}

ov_status_e ov_dimensions_create(ov_dimensions_t** dimensions) {
    if (!dimensions) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *dimensions = nullptr;
    try {
        std::unique_ptr<ov_dimensions_t> dims(new ov_dimensions_t);
        *dimensions = dims.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_dimensions_add(ov_dimensions_t* dimensions, int64_t value) {
    if (!dimensions || value < 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    dimensions->object.emplace_back(value);
    return ov_status_e::OK;
}

ov_status_e ov_dimensions_add_dynamic(ov_dimensions_t* dimensions, int64_t min_dimension, int64_t max_dimension) {
    if (!dimensions || min_dimension < -1 || max_dimension < -1) {
        return ov_status_e::INVALID_C_PARAM;
    }
    dimensions->object.emplace_back(min_dimension, max_dimension);
    return ov_status_e::OK;
}

void ov_dimensions_free(ov_dimensions_t* dimensions) {
    if (dimensions)
        delete dimensions;
}
