// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_tensor.h"

#include "common.h"

const std::map<ov_element_type_e, ov::element::Type> element_type_map = {
    {ov_element_type_e::UNDEFINED, ov::element::dynamic},
    {ov_element_type_e::DYNAMIC, ov::element::dynamic},
    {ov_element_type_e::BOOLEAN, ov::element::boolean},
    {ov_element_type_e::BF16, ov::element::bf16},
    {ov_element_type_e::F16, ov::element::f16},
    {ov_element_type_e::F32, ov::element::f32},
    {ov_element_type_e::F64, ov::element::f64},
    {ov_element_type_e::I4, ov::element::i4},
    {ov_element_type_e::I8, ov::element::i8},
    {ov_element_type_e::I16, ov::element::i16},
    {ov_element_type_e::I32, ov::element::i32},
    {ov_element_type_e::I64, ov::element::i64},
    {ov_element_type_e::U1, ov::element::u1},
    {ov_element_type_e::U2, ov::element::u2},
    {ov_element_type_e::U3, ov::element::u3},
    {ov_element_type_e::U4, ov::element::u4},
    {ov_element_type_e::U6, ov::element::u6},
    {ov_element_type_e::U8, ov::element::u8},
    {ov_element_type_e::U16, ov::element::u16},
    {ov_element_type_e::U32, ov::element::u32},
    {ov_element_type_e::U64, ov::element::u64},
    {ov_element_type_e::NF4, ov::element::nf4},
    {ov_element_type_e::F8E4M3, ov::element::f8e4m3},
    {ov_element_type_e::F8E5M3, ov::element::f8e5m2},
    {ov_element_type_e::STRING, ov::element::string},
    {ov_element_type_e::F4E2M1, ov::element::f4e2m1},
    {ov_element_type_e::F8E8M0, ov::element::f8e8m0},
};

ov_element_type_e find_ov_element_type_e(ov::element::Type type) {
    for (auto iter = element_type_map.begin(); iter != element_type_map.end(); iter++) {
        if (iter->second == type) {
            return iter->first;
        }
    }
    return ov_element_type_e::UNDEFINED;
}

ov::element::Type get_element_type(ov_element_type_e type) {
    return element_type_map.at(type);
}

ov_status_e ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor) {
    if (!tensor || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        auto tmp_type = get_element_type(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        _tensor->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape);
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                                           const ov_shape_t shape,
                                           void* host_ptr,
                                           ov_tensor_t** tensor) {
    if (!tensor || !host_ptr || element_type_map.find(type) == element_type_map.end()) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_tensor_t> _tensor(new ov_tensor_t);
        auto tmp_type = get_element_type(type);
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        _tensor->object = std::make_shared<ov::Tensor>(tmp_type, tmp_shape, host_ptr);
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape) {
    if (!tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        tensor->object->set_shape(tmp_shape);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape) {
    if (!tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto tmp_shape = tensor->object->get_shape();
        ov_shape_create(tmp_shape.size(), nullptr, shape);
        std::copy_n(tmp_shape.begin(), tmp_shape.size(), shape->dims);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type) {
    if (!tensor || !type) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto tmp_type = tensor->object->get_element_type();
        *type = find_ov_element_type_e(tmp_type);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size) {
    if (!tensor || !elements_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *elements_size = tensor->object->get_size();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size) {
    if (!tensor || !byte_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *byte_size = tensor->object->get_byte_size();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_tensor_data(const ov_tensor_t* tensor, void** data) {
    if (!tensor || !data) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *data = tensor->object->data();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_tensor_free(ov_tensor_t* tensor) {
    if (tensor)
        delete tensor;
}
