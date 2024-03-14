// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <string>
#include <iterator>
#include <climits>

#include "Python.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/pass/serialize.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/graph/ops/constant.hpp"
#include "pyopenvino/core/infer_request.hpp"

namespace py = pybind11;

namespace Common {

namespace containers {
    using TensorIndexMap = std::map<size_t, ov::Tensor>;

    const TensorIndexMap cast_to_tensor_index_map(const py::dict& inputs);
}; // namespace containers

namespace values {

// Minimum amount of bits for common numpy types. Used to perform checks against OV types.
constexpr size_t min_bitwidth = sizeof(int8_t) * CHAR_BIT;

}; // namespace values

// Helpers for dtypes and OpenVINO types
namespace type_helpers {

const std::map<ov::element::Type, py::dtype>& ov_type_to_dtype();

py::dtype get_dtype(const ov::element::Type& ov_type);

const std::map<std::string, ov::element::Type>& dtype_to_ov_type();

ov::element::Type get_ov_type(const py::array& array);

ov::element::Type get_ov_type(py::dtype& dtype);
}

// Helpers for string types and numpy arrays of strings
namespace string_helpers {

py::array bytes_array_from_tensor(ov::Tensor&& t);

py::array string_array_from_tensor(ov::Tensor&& t);

void fill_tensor_from_bytes(ov::Tensor& tensor, py::array& array);

void fill_tensor_from_strings(ov::Tensor& tensor, py::array& array);

void fill_string_tensor_data(ov::Tensor& tensor, py::array& array);

}; // namespace string_helpers

// Helpers for numpy arrays
namespace array_helpers {

bool is_contiguous(const py::array& array);

std::vector<size_t> get_shape(const py::array& array);

std::vector<size_t> get_strides(const py::array& array);

py::array as_contiguous(py::array& array, ov::element::Type type);

template <typename T>
py::array array_from_tensor_t(ov::Tensor&& t, py::dtype&& dtype) {
    auto tmp = ov::op::v0::Constant(t).cast_vector<T>();
    auto array = py::array(dtype, tmp.size(), tmp.data());
    return array.reshape(t.get_shape());
}

py::array array_from_tensor(ov::Tensor&& t, bool is_shared);

py::array array_from_tensor(ov::Tensor&& t, const ov::element::Type& dst_dtype);

py::array array_from_constant(ov::op::v0::Constant&& c, bool is_shared);

template <typename T>
std::vector<T> array_as_vector(py::array& array){
    T *ptr = static_cast<T*>(const_cast<void*>(array.data()));
    return std::vector<T>(ptr, ptr + array.size());
}

template <typename T>
std::vector<char> array_as_vector_bool(py::array& array) {
    std::vector<char> result;
    result.reserve(array.size());

    for(long int i = 0; i < array.size(); i++) {
        result.emplace_back(*(static_cast<T*>(const_cast<void*>(array.data())) + i) != 0 ? 1 : 0);
    }

    return result;
}
}; // namespace array_helpers

namespace tensor_helpers {

template <typename T>
void fill_tensor_t(ov::Tensor& tensor, py::array& array, ov::element::Type& dst_dtype) {
    if (dst_dtype == ov::element::boolean) {
        auto tmp = ov::op::v0::Constant(dst_dtype,
                                        array_helpers::get_shape(array),
                                        array_helpers::array_as_vector_bool<T>(array));
        std::memcpy(tensor.data(), tmp.get_data_ptr(), tmp.get_byte_size());
    } else {
        auto tmp = ov::op::v0::Constant(dst_dtype,
                                        array_helpers::get_shape(array),
                                        array_helpers::array_as_vector<T>(array));
        std::memcpy(tensor.data(), tmp.get_data_ptr(), tmp.get_byte_size());
    }
}

void fill_tensor(ov::Tensor& tensor, py::array& array);

void fill_tensor(ov::Tensor& tensor, py::array& array, ov::element::Type& dst_dtype);
}; // namespace tensor_helpers

namespace constant_helpers {
template <typename T>
std::vector<size_t> _get_byte_strides(const ov::Shape& s) {
    std::vector<size_t> byte_strides;
    std::vector<size_t> element_strides = ov::row_major_strides(s);
    for (auto v : element_strides) {
        byte_strides.push_back(static_cast<size_t>(v) * sizeof(T));
    }
    return byte_strides;
}

std::vector<size_t> _get_strides(const ov::op::v0::Constant& self);
}; // namespace constant_helpers


template <typename T>
T create_copied(py::array& array);

template <typename T>
T create_copied(ov::Tensor& array);

template <typename T>
T create_copied(py::array& array, ov::element::Type& dst_dtype);

template <typename T>
T create_shared(py::array& array);

template <typename T>
T create_shared(ov::Tensor& array);

template <typename T>
T create_shared(py::array& array, ov::element::Type& dst_dtype);

template <typename T, typename D>
T object_from_data(D& data, bool shared_memory) {
    if (shared_memory) {
        return create_shared<T>(data);
    }
    return create_copied<T>(data);
}

template <typename T, typename D>
T object_from_data(D& data, ov::element::Type& dst_dtype, bool shared_memory) {
    if (shared_memory) {
        return create_shared<T>(data, dst_dtype);
    }
    return create_copied<T>(data, dst_dtype);
}

ov::Tensor tensor_from_pointer(py::array& array, const ov::Shape& shape, const ov::element::Type& ov_type);

ov::Tensor tensor_from_pointer(py::array& array, const ov::Output<const ov::Node>& port);

ov::PartialShape partial_shape_from_list(const py::list& shape);

const ov::Tensor& cast_to_tensor(const py::handle& tensor);

void set_request_tensors(ov::InferRequest& request, const py::dict& inputs);

uint32_t get_optimal_number_of_requests(const ov::CompiledModel& actual);

py::dict outputs_to_dict(InferRequestWrapper& request, bool share_outputs, bool decode_strings);

ov::pass::Serialize::Version convert_to_version(const std::string& version);

template <typename T>
std::string get_class_name(const T& obj) {
    return py::str(py::cast(obj).get_type().attr("__name__"));
}

template <typename T>
std::string get_simple_repr(const T& obj) {
    std::string class_name = get_class_name(obj);
    return "<" + class_name + ">";
}

// Use only with classes that are not creatable by users on Python's side, because
// Objects created in Python that are wrapped with such wrapper will cause memory leaks.
template <typename T>
class ref_wrapper {
    std::reference_wrapper<T> impl;

public:
    explicit ref_wrapper(T* p) : impl(*p) {}
    T* get() const {
        return &impl.get();
    }
};

namespace docs {
template<typename Container, typename std::enable_if<std::is_same<typename Container::value_type, std::string>::value, bool>::type = true>
std::string container_to_string(const Container& c, const std::string& delimiter) {
    if (c.size() == 0) {
    	return std::string{};
    }

    std::string buffer;
    for (const auto& elem : c) {
        buffer += elem + delimiter;
    }

    buffer.erase(buffer.end() - delimiter.size(), buffer.end());

    return buffer;
}

template<typename Container, typename std::enable_if<!std::is_same<typename Container::value_type, std::string>::value, bool>::type = true>
std::string container_to_string(const Container& c, const std::string& delimiter) {
    if (c.size() == 0) {
    	return std::string{};
    }

    std::string buffer;
    for (const auto& elem : c) {
        buffer += py::cast<std::string>(py::cast(elem).attr("__repr__")()) + delimiter;
    }

    buffer.erase(buffer.end() - delimiter.size(), buffer.end());

    return buffer;
}
};  // namespace docs

};  // namespace Common
