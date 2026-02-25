// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_partial_shape.h"

#include "common.h"

inline bool check_dimension(const ov_dimension_t* dims, int64_t size) {
    for (auto i = 0; i < size; i++) {
        auto& _dim = dims[i];
        if (_dim.max < -1 || _dim.min < -1 || _dim.max < _dim.min)
            return false;
    }
    return true;
}

ov_status_e ov_partial_shape_create(const int64_t rank,
                                    const ov_dimension_t* dims,
                                    ov_partial_shape_t* partial_shape_obj) {
    if (!partial_shape_obj || rank <= 0 || !dims || !check_dimension(dims, rank)) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_dimension_t> _dims(new ov_dimension_t[rank]);
        partial_shape_obj->dims = _dims.release();
        std::memcpy(partial_shape_obj->dims, dims, rank * sizeof(ov_dimension_t));
        partial_shape_obj->rank.max = rank;
        partial_shape_obj->rank.min = rank;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_partial_shape_create_dynamic(const ov_rank_t rank,
                                            const ov_dimension_t* dims,
                                            ov_partial_shape_t* partial_shape_obj) {
    if (!partial_shape_obj || rank.min < -1 || rank.max < -1 || rank.min > rank.max) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        partial_shape_obj->rank = rank;
        if (ov_rank_is_dynamic(rank)) {
            // Dynamic rank
            partial_shape_obj->dims = nullptr;
        } else {
            // Static rank
            if (!dims || !check_dimension(dims, rank.max)) {
                return ov_status_e::INVALID_C_PARAM;
            }
            auto size = rank.max;
            std::unique_ptr<ov_dimension_t> _dims(new ov_dimension_t[size]);
            partial_shape_obj->dims = _dims.release();
            std::memcpy(partial_shape_obj->dims, dims, size * sizeof(ov_dimension_t));
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_partial_shape_create_static(const int64_t rank,
                                           const int64_t* dims,
                                           ov_partial_shape_t* partial_shape_obj) {
    if (!partial_shape_obj || rank < 0 || !dims) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_dimension_t> _dims(new ov_dimension_t[rank]);
        partial_shape_obj->dims = _dims.release();
        partial_shape_obj->rank.max = rank;
        partial_shape_obj->rank.min = rank;
        for (auto i = 0; i < rank; i++) {
            if (dims[i] <= 0) {
                return ov_status_e::INVALID_C_PARAM;
            }
            partial_shape_obj->dims[i].min = dims[i];
            partial_shape_obj->dims[i].max = dims[i];
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_partial_shape_free(ov_partial_shape_t* partial_shape) {
    if (partial_shape && partial_shape->dims)
        delete[] partial_shape->dims;
}

ov_status_e ov_partial_shape_to_shape(const ov_partial_shape_t partial_shape, ov_shape_t* shape) {
    if (!shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        if (ov_rank_is_dynamic(partial_shape.rank)) {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        auto rank = partial_shape.rank.max;
        ov_shape_create(rank, nullptr, shape);

        for (auto i = 0; i < rank; ++i) {
            auto& ov_dim = partial_shape.dims[i];
            if (!ov_dimension_is_dynamic(ov_dim))
                shape->dims[i] = ov_dim.max;
            else
                return ov_status_e::PARAMETER_MISMATCH;
        }
        shape->rank = rank;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_shape_to_partial_shape(const ov_shape_t shape, ov_partial_shape_t* partial_shape) {
    if (!partial_shape || shape.rank <= 0 || !shape.dims) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        partial_shape->rank.min = shape.rank;
        partial_shape->rank.max = shape.rank;
        auto size = shape.rank;
        std::unique_ptr<ov_dimension_t> _dims(new ov_dimension_t[size]);
        partial_shape->dims = _dims.release();
        for (auto i = 0; i < size; i++) {
            partial_shape->dims[i].min = shape.dims[i];
            partial_shape->dims[i].max = shape.dims[i];
        }
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

bool ov_partial_shape_is_dynamic(const ov_partial_shape_t partial_shape) {
    if (ov_rank_is_dynamic(partial_shape.rank)) {
        return true;
    }
    auto rank = partial_shape.rank.max;
    for (auto i = 0; i < rank; i++) {
        auto dim = partial_shape.dims[i];
        if (ov_dimension_is_dynamic(dim)) {
            return true;
        }
    }
    return false;
}

const char* ov_partial_shape_to_string(const ov_partial_shape_t partial_shape) {
    // Dynamic rank
    if (ov_rank_is_dynamic(partial_shape.rank)) {
        return str_to_char_array("?");
    }

    // Static rank
    auto rank = partial_shape.rank.max;
    std::string str = std::string("{");
    for (auto i = 0; i < rank; i++) {
        auto _dim = &partial_shape.dims[i];
        ov::Dimension item(_dim->min, _dim->max);
        std::ostringstream out;
        out.str("");
        out << item;
        str += out.str();
        if (i < rank - 1)
            str += ",";
    }
    str += std::string("}");
    const char* res = str_to_char_array(str);

    return res;
}
