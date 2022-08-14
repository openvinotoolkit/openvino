// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_shape.h"

#include "common.h"

ov_status_e ov_shape_init(ov_shape_t* shape, int64_t rank) {
    if (!shape || rank <= 0) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<int64_t> _dims(new int64_t[rank]);
        shape->rank = rank;
        shape->dims = _dims.release();
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_shape_deinit(ov_shape_t* shape) {
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