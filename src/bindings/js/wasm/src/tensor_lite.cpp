// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/tensor_lite.h"

#include <stdio.h>

#include <iostream>

#include "../include/shape_lite.h"
#include "openvino/openvino.hpp"

using supported_type_t = std::unordered_map<std::string, ov::element::Type>;
ov::element::Type getType(std::string value, const supported_type_t& supported_precisions) {
    const auto precision = supported_precisions.find(value);
    if (precision == supported_precisions.end()) {
        throw std::logic_error("\"" + value + "\"" + " is not a valid precision");
    }

    return precision->second;
}
ov::element::Type get_type(const std::string& value) {
    static const supported_type_t supported_types = {
        {"i8", ov::element::i8},
        {"u8", ov::element::u8},
        // {"u8c", ov::element::u8},

        {"i16", ov::element::i16},
        {"u16", ov::element::u16},

        {"i32", ov::element::i32},
        {"u32", ov::element::u32},

        {"f32", ov::element::f32},
        {"f64", ov::element::f64},

        {"i64", ov::element::i64},
        {"u64", ov::element::u64},
    };

    return getType(value, supported_types);
}

template <typename T>
uintptr_t get_data_by_tensor(ov::Tensor tensor) {
    auto type = tensor.get_element_type();
    T* arr = reinterpret_cast<T*>(tensor.data(type));

    return uintptr_t(&arr[0]);
}

TensorLite::TensorLite(ov::element::Type type, uintptr_t data_buffer, ShapeLite* shape) {
    this->tensor = ov::Tensor(type, shape->get_original(), &data_buffer);
}

TensorLite::TensorLite(std::string type_str, uintptr_t data_buffer, ShapeLite* shape) {
    try {
        ov::element::Type type = get_type(type_str);

        if (type == ov::element::u8) {
            auto arr = reinterpret_cast<uint8_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::i8) {
            auto arr = reinterpret_cast<int8_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::u16) {
            auto arr = reinterpret_cast<uint16_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::i16) {
            auto arr = reinterpret_cast<int16_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::u32) {
            auto arr = reinterpret_cast<uint32_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::i32) {
            auto arr = reinterpret_cast<uint32_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::f32) {
            auto arr = reinterpret_cast<float*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::f64) {
            auto arr = reinterpret_cast<double*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::u64) {
            auto arr = reinterpret_cast<uint64_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
        if (type == ov::element::i64) {
            auto arr = reinterpret_cast<int64_t*>(data_buffer);
            this->tensor = ov::Tensor(type, shape->get_original(), &arr[0]);
        }
    } catch (const std::exception& e) {
        std::cout << "== Error in Tensor constructor: " << e.what() << std::endl;
        throw e;
    }
}

TensorLite::TensorLite(const ov::Tensor& tensor) {
    ov::Shape originalShape = tensor.get_shape();

    this->tensor = tensor;
}

ShapeLite* TensorLite::get_shape() {
    ov::Shape originalShape = tensor.get_shape();

    return new ShapeLite(&originalShape);
}

uintptr_t TensorLite::get_data() {
    // FIXME: adapt data to tensor type
    return get_data_by_tensor<float>(this->tensor);
}

std::string TensorLite::get_precision() {
    ov::element::Type type = this->tensor.get_element_type();

    return type.get_type_name();
}

ov::Tensor* TensorLite::get_tensor() {
    return &this->tensor;
}
