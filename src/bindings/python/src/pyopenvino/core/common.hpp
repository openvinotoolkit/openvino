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

py::array array_from_tensor(ov::Tensor&& t, bool is_shared);

}; // namespace array_helpers

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

// See #18388
template <typename T>
class ModelHolder {
    static_assert(std::is_same<T, ov::Model>::value, "ModelHolder can only hold ov::Model");
    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::CompiledModel> m_compiled_model;

public:
    ModelHolder() = default;
    ModelHolder(const ModelHolder&) = default;
    ModelHolder(ModelHolder&&) = default;
    ModelHolder& operator=(const ModelHolder&) = default;
    ModelHolder& operator=(ModelHolder&&) = default;

    // construct from std::shared_ptr<ov::Model>
    ModelHolder(const std::shared_ptr<ov::Model>& model) : m_model(model) {}
    ModelHolder(std::shared_ptr<ov::Model>&& model) : m_model(std::move(model)) {}

    // special constructor for CompiledModel::get_runtime_model()
    ModelHolder(std::shared_ptr<ov::Model>&& model, std::shared_ptr<ov::CompiledModel>& compiled_model)
        : m_model(std::move(model)),
          m_compiled_model(compiled_model) {}

    // calls shared_from_this() automatically by the constructor of std::shared_ptr
    ModelHolder(ov::Model* model) : m_model(model) {}

    // make sure the compiled model is destructed after the runtime model to
    // keep the dynamic-loaded library alive, as described in issue #18388
    ~ModelHolder() noexcept {
        m_model.reset();
    }

    // required by pybind11
    ov::Model* get() const noexcept {
        return m_model.get();
    }

    const std::shared_ptr<ov::Model>& get_shared() const noexcept {
        return m_model;
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

PYBIND11_DECLARE_HOLDER_TYPE(T, Common::ModelHolder<T>);

// Specialization of type caster for std::shared_ptr<ov::Model>, to cast
// std::shared_ptr<ov::Model> to Common::ModelHolder<ov::Model> and vice versa.
// Should be included in every file that uses std::shared_ptr<ov::Model> as
// args or return value in pybind11 bindings.
namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
class type_caster<std::shared_ptr<ov::Model>> {
    using base = type_caster<Common::ModelHolder<ov::Model>>;
    base m_base;

public:
    PYBIND11_TYPE_CASTER(std::shared_ptr<ov::Model>, const_name<std::shared_ptr<ov::Model>>());
    bool load(handle src, bool convert) {
        if (!m_base.load(src, convert))
            return false;
        value = static_cast<Common::ModelHolder<ov::Model>>(m_base).get_shared();
        return true;
    }
    static handle cast(const std::shared_ptr<ov::Model>& src, return_value_policy policy, handle parent) {
        return base::cast(Common::ModelHolder<ov::Model>(src), policy, parent);
    }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
