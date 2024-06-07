// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_node.h"

#include "common.h"

ov_status_e ov_const_port_get_shape(const ov_output_const_port_t* port, ov_shape_t* tensor_shape) {
    if (!port || !tensor_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto shape = port->object->get_shape();
        ov_shape_create(shape.size(), nullptr, tensor_shape);
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_port_get_shape(const ov_output_port_t* port, ov_shape_t* tensor_shape) {
    if (!port || !tensor_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto shape = port->object->get_shape();
        ov_shape_create(shape.size(), nullptr, tensor_shape);
        std::copy_n(shape.begin(), shape.size(), tensor_shape->dims);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_port_get_any_name(const ov_output_const_port_t* port, char** tensor_name) {
    if (!port || !tensor_name) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        *tensor_name = str_to_char_array(port->object->get_any_name());
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_port_get_partial_shape(const ov_output_const_port_t* port, ov_partial_shape_t* partial_shape) {
    if (!port || !partial_shape) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto pshape = port->object->get_partial_shape();
        auto rank = pshape.rank();

        partial_shape->rank.min = rank.get_min_length();
        partial_shape->rank.max = rank.get_max_length();
        if (rank.is_dynamic()) {
            partial_shape->dims = nullptr;
        } else {
            auto size = rank.get_length();
            if (static_cast<size_t>(size) != pshape.size()) {
                return ov_status_e::PARAMETER_MISMATCH;
            }
            std::unique_ptr<ov_dimension_t> _dimensions(new ov_dimension_t[size]);
            partial_shape->dims = _dimensions.release();
            auto iter = pshape.begin();
            for (auto i = 0; iter != pshape.end(); iter++, i++) {
                partial_shape->dims[i].min = iter->get_min_length();
                partial_shape->dims[i].max = iter->get_max_length();
            }
        }
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

ov_status_e ov_port_get_element_type(const ov_output_const_port_t* port, ov_element_type_e* tensor_type) {
    if (!port) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto type = (ov::element::Type_t)port->object->get_element_type();
        *tensor_type = find_ov_element_type_e(type);
    }
    CATCH_OV_EXCEPTIONS

    return ov_status_e::OK;
}

void ov_output_port_free(ov_output_port_t* port) {
    if (port)
        delete port;
}

void ov_output_const_port_free(ov_output_const_port_t* port) {
    if (port)
        delete port;
}
