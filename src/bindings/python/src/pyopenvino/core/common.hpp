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

std::map<int, ov::element::Type> init_num_to_ov_type();

const std::map<int, ov::element::Type>& dtype_num_to_ov_type();

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

py::array array_from_tensor(ov::Tensor&& t, bool is_shared);

template <typename T>
py::array array_from_constant_cast_bool(ov::op::v0::Constant&& c, py::dtype& dst_dtype) {
    std::vector<char> result;
    size_t size = c.get_byte_size() / sizeof(T);

    result.reserve(size);

    for(size_t i = 0; i < size; i++) {
        result.emplace_back(*(static_cast<const T*>(c.get_data_ptr()) + i) != 0 ? 1 : 0);
    }

    return py::array(dst_dtype, c.get_shape(), result.data());
}

template <typename T>
py::array array_from_constant_cast(ov::op::v0::Constant&& c, py::dtype& dst_dtype) {
    auto tmp = c.cast_vector<T>();
    return py::array(dst_dtype, c.get_shape(), tmp.data());
}

py::array array_from_constant_copy(ov::op::v0::Constant&& c);

py::array array_from_constant_copy(ov::op::v0::Constant&& c, py::dtype& dst_dtype);

py::array array_from_constant_view(ov::op::v0::Constant&& c);

}; // namespace array_helpers

namespace constant_helpers {
std::vector<size_t> _get_byte_strides(const ov::Shape& s, size_t element_byte_size);

template <typename T>
std::vector<size_t> _get_byte_strides(const ov::Shape& s) {
    return _get_byte_strides(s, sizeof(T));
}

std::vector<size_t> _get_strides(const ov::op::v0::Constant& self);

}; // namespace constant_helpers

// Helpers for shapes
namespace shape_helpers {

template <typename T>
void get_slice(T& result, const T& shape, size_t start, const size_t step, const size_t slicelength) {
    for (size_t i = 0; i < slicelength; ++i) {
        result[i] = shape[start];
        start += step;
    }
}

}; // namespace shape_helpers

template <typename T>
T create_copied(py::array& array);

template <typename T>
T create_copied(ov::Tensor& array);

template <typename T>
T create_shared(py::array& array);

template <typename T>
T create_shared(ov::Tensor& array);

template <typename T, typename D>
T object_from_data(D& data, bool shared_memory) {
    if (shared_memory) {
        return create_shared<T>(data);
    }
    return create_copied<T>(data);
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
