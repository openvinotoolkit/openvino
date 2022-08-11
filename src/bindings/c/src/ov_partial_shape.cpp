// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_partial_shape.h"

#include "common.h"

ov_status_e ov_partial_shape_create(ov_partial_shape_t** partial_shape_obj, ov_rank_t* rank, ov_dimensions_t* dims) {
    if (!partial_shape_obj || !rank) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *partial_shape_obj = nullptr;
    try {
        std::unique_ptr<ov_partial_shape_t> partial_shape(new ov_partial_shape_t);
        if (rank->object.is_dynamic()) {
            partial_shape->rank = rank->object;
        } else {
            if (rank->object.get_length() != dims->object.size()) {
                return ov_status_e::INVALID_C_PARAM;
            }
            partial_shape->rank = rank->object;
            partial_shape->dims = dims->object;
        }
        *partial_shape_obj = partial_shape.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_partial_shape_free(ov_partial_shape_t* partial_shape) {
    if (partial_shape)
        delete partial_shape;
}

const char* ov_partial_shape_to_string(ov_partial_shape_t* partial_shape) {
    if (!partial_shape) {
        return str_to_char_array("Error: null partial_shape!");
    }

    // dynamic rank
    if (partial_shape->rank.is_dynamic()) {
        return str_to_char_array("?");
    }

    // static rank
    auto rank = partial_shape->rank.get_length();
    if (rank != partial_shape->dims.size()) {
        return str_to_char_array("rank error");
    }
    std::string str = std::string("{");
    int i = 0;
    for (auto& item : partial_shape->dims) {
        std::ostringstream out;
        out.str("");
        out << item;
        str += out.str();
        if (i++ < rank - 1)
            str += ",";
    }
    str += std::string("}");
    const char* res = str_to_char_array(str);

    return res;
}

ov_status_e ov_partial_shape_to_shape(ov_partial_shape_t* partial_shape, ov_shape_t* shape) {
    if (!partial_shape || !shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        if (partial_shape->rank.is_dynamic()) {
            return ov_status_e::PARAMETER_MISMATCH;
        }
        auto rank = partial_shape->rank.get_length();
        ov_shape_init(shape, rank);

        for (auto i = 0; i < rank; ++i) {
            auto& ov_dim = partial_shape->dims[i];
            if (ov_dim.is_static())
                shape->dims[i] = ov_dim.get_length();
            else
                return ov_status_e::PARAMETER_MISMATCH;
        }
        shape->rank = rank;
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_shape_to_partial_shape(ov_shape_t* shape, ov_partial_shape_t** partial_shape) {
    if (!partial_shape || !shape || shape->rank <= 0 || !shape->dims) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::unique_ptr<ov_partial_shape_t> _partial_shape(new ov_partial_shape_t);
        _partial_shape->rank = ov::Dimension(shape->rank);
        for (int i = 0; i < shape->rank; i++) {
            _partial_shape->dims.emplace_back(shape->dims[i]);
        }
        *partial_shape = _partial_shape.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}
