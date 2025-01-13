// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_shape.h"

#include "common.h"

inline bool check_shape_dimension(const int64_t* dims, const int64_t size) {
    for (auto i = 0; i < size; i++) {
        if (dims[i] < 0) {
            return false;
        }
    }
    return true;
}

ov_status_e ov_shape_create(const int64_t rank, const int64_t* dims, ov_shape_t* shape) {
    if (!shape || rank <= 0 || (dims && !check_shape_dimension(dims, rank))) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<int64_t> _dims(new int64_t[rank]);
        shape->dims = _dims.release();
        if (dims) {
            std::memcpy(shape->dims, dims, rank * sizeof(int64_t));
        }
        shape->rank = rank;
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_shape_free(ov_shape_t* shape) {
    if (!shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    shape->rank = 0;
    if (shape->dims) {
        delete[] shape->dims;
        shape->dims = nullptr;
    }

    return ov_status_e::OK;
}
