// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <openvino/core/type/element_type.hpp>
#include <string>
#include <iterator>

#include "Python.h"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/pass/serialize.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/graph/any.hpp"

namespace py = pybind11;

namespace Common {
const std::map<ov::element::Type, py::dtype>& ov_type_to_dtype();

const std::map<std::string, ov::element::Type>& dtype_to_ov_type();

ov::Tensor tensor_from_pointer(py::array& array, const ov::Shape& shape, const ov::element::Type& ov_type);

ov::Tensor tensor_from_numpy(py::array& array, bool shared_memory);

ov::PartialShape partial_shape_from_list(const py::list& shape);

ov::PartialShape partial_shape_from_str(const std::string& value);

ov::Dimension dimension_from_str(const std::string& value);

py::array as_contiguous(py::array& array, ov::element::Type type);

const ov::Tensor& cast_to_tensor(const py::handle& tensor);

const Containers::TensorNameMap cast_to_tensor_name_map(const py::dict& inputs);

const Containers::TensorIndexMap cast_to_tensor_index_map(const py::dict& inputs);

void set_request_tensors(ov::InferRequest& request, const py::dict& inputs);

uint32_t get_optimal_number_of_requests(const ov::CompiledModel& actual);

py::dict outputs_to_dict(const std::vector<ov::Output<const ov::Node>>& outputs, ov::InferRequest& request);

ov::pass::Serialize::Version convert_to_version(const std::string& version);

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
