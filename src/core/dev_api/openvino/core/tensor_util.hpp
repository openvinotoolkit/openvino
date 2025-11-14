#pragma once

#include <optional>
#include <stdexcept>
#include <cstring>
#include <type_traits>
#include <vector>
#include <string>

#include "openvino/op/constant.hpp"

namespace ov {
namespace util {

OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const ov::Tensor& rhs);
template <typename T>
OPENVINO_API Tensor greater_equal(const ov::Tensor& lhs, const T& element);
OPENVINO_API bool reduce_and(const ov::Tensor& t);
template <typename T>
OPENVINO_API std::optional<std::vector<T>> to_vector(const ov::Tensor& t);

inline bool is_supported_element_type(const element::Type_t& et) {
    switch (et) {
        case element::f32:
        case element::f16:
        case element::bf16:
        case element::i64:
        case element::i32:
        case element::i16:
        case element::i8:
        case element::u64:
        case element::u32:
        case element::u16:
        case element::u8:
        case element::boolean:
            return true;
        default:
            return false;
    }
}


template <typename T>
Tensor make_tensor_of_value(const element::Type_t& et, const T& value, Shape shape = {}) {
    if (!is_supported_element_type(et)) {
        throw std::runtime_error("Unsupported element type in make_tensor_of_value for numeric template."); //this will throw a message if it is run with string
    }

    // Create a Constant node with the requested element type and value
    auto c = op::v0::Constant(et, shape, value);
    auto t = Tensor(et, shape);

    // Safety checks
    if (!c.get_data_ptr()) {
        throw std::runtime_error("Constant returned a null data pointer in make_tensor_of_value.");
    }
    if (!t.data()) {
        throw std::runtime_error("Target tensor has null data pointer in make_tensor_of_value.");
    }

    std::memcpy(t.data(), c.get_data_ptr(), t.get_byte_size());
    return t;
}


inline Tensor make_tensor_of_value(const element::Type_t& /*et*/, const std::string& value, Shape shape = {}) {
    std::vector<uint8_t> bytes(value.begin(), value.end());

    if (bytes.empty()) {
        Shape zero_shape = shape.empty() ? Shape{0} : shape;
        auto c = op::v0::Constant(element::u8, zero_shape, std::vector<uint8_t>{});
        Tensor t(element::u8, zero_shape);
        return t;
    }

   
    if (shape.empty()) {
        shape = {bytes.size()};
    }

    auto c = op::v0::Constant(element::u8, shape, bytes);
    auto t = Tensor(element::u8, shape);

    if (!c.get_data_ptr() || !t.data()) {
        throw std::runtime_error("Invalid data pointer when creating u8 tensor from string.");
    }

    std::memcpy(t.data(), c.get_data_ptr(), t.get_byte_size());
    return t;
}


inline Tensor make_tensor_of_value(const element::Type_t& et, const char* cstr, Shape shape = {}) {
    if (!cstr) {
        return make_tensor_of_value(et, std::string{}, shape);
    }
    return make_tensor_of_value(et, std::string(cstr), shape);
}


template <typename T>
Tensor greater_equal(const ov::Tensor& lhs, const T& element) {
    if (!lhs)
        return {};
    const auto& other = make_tensor_of_value(lhs.get_element_type(), element);
    return greater_equal(lhs, other);
}

template <typename T>
std::optional<std::vector<T>> to_vector(const ov::Tensor& t) {
    std::optional<std::vector<T>> result;
    if (t)
        result = ov::op::v0::Constant(t).cast_vector<T>();
    return result;
}

}  
}
