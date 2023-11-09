// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

#include <unordered_map>

#include "Python.h"
#include "openvino/core/except.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"

#define C_CONTIGUOUS py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_

namespace Common {
const std::map<ov::element::Type, py::dtype>& ov_type_to_dtype() {
    static const std::map<ov::element::Type, py::dtype> ov_type_to_dtype_mapping = {
        {ov::element::f16, py::dtype("float16")},
        {ov::element::bf16, py::dtype("float16")},
        {ov::element::f32, py::dtype("float32")},
        {ov::element::f64, py::dtype("float64")},
        {ov::element::i8, py::dtype("int8")},
        {ov::element::i16, py::dtype("int16")},
        {ov::element::i32, py::dtype("int32")},
        {ov::element::i64, py::dtype("int64")},
        {ov::element::u8, py::dtype("uint8")},
        {ov::element::u16, py::dtype("uint16")},
        {ov::element::u32, py::dtype("uint32")},
        {ov::element::u64, py::dtype("uint64")},
        {ov::element::boolean, py::dtype("bool")},
        {ov::element::u1, py::dtype("uint8")},
        {ov::element::u4, py::dtype("uint8")},
        {ov::element::nf4, py::dtype("uint8")},
        {ov::element::i4, py::dtype("int8")},
    };
    return ov_type_to_dtype_mapping;
}

const std::map<std::string, ov::element::Type>& dtype_to_ov_type() {
    static const std::map<std::string, ov::element::Type> dtype_to_ov_type_mapping = {
        {"float16", ov::element::f16},
        {"float32", ov::element::f32},
        {"float64", ov::element::f64},
        {"int8", ov::element::i8},
        {"int16", ov::element::i16},
        {"int32", ov::element::i32},
        {"int64", ov::element::i64},
        {"uint8", ov::element::u8},
        {"uint16", ov::element::u16},
        {"uint32", ov::element::u32},
        {"uint64", ov::element::u64},
        {"bool", ov::element::boolean},
    };
    return dtype_to_ov_type_mapping;
}

namespace containers {
const TensorIndexMap cast_to_tensor_index_map(const py::dict& inputs) {
    TensorIndexMap result_map;
    for (auto&& input : inputs) {
        int idx;
        if (py::isinstance<py::int_>(input.first)) {
            idx = input.first.cast<int>();
        } else {
            throw py::type_error("incompatible function arguments!");
        }
        if (py::isinstance<ov::Tensor>(input.second)) {
            auto tensor = Common::cast_to_tensor(input.second);
            result_map[idx] = tensor;
        } else {
            OPENVINO_THROW("Unable to cast tensor " + std::to_string(idx) + "!");
        }
    }
    return result_map;
}
};  // namespace containers

namespace array_helpers {

bool is_contiguous(const py::array& array) {
    return C_CONTIGUOUS == (array.flags() & C_CONTIGUOUS);
}

ov::element::Type get_ov_type(const py::array& array) {
    return Common::dtype_to_ov_type().at(py::str(array.dtype()));
}

std::vector<size_t> get_shape(const py::array& array) {
    return std::vector<size_t>(array.shape(), array.shape() + array.ndim());
}

std::vector<size_t> get_strides(const py::array& array) {
    return std::vector<size_t>(array.strides(), array.strides() + array.ndim());
}

py::array as_contiguous(py::array& array, ov::element::Type type) {
    switch (type) {
    // floating
    case ov::element::f64:
        return array.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    case ov::element::f32:
        return array.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    // signed
    case ov::element::i64:
        return array.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::i32:
        return array.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::i16:
        return array.cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::i8:
        return array.cast<py::array_t<int8_t, py::array::c_style | py::array::forcecast>>();
    // unsigned
    case ov::element::u64:
        return array.cast<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::u32:
        return array.cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::u16:
        return array.cast<py::array_t<uint16_t, py::array::c_style | py::array::forcecast>>();
    case ov::element::u8:
        return array.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    // other
    case ov::element::boolean:
        return array.cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>();
    case ov::element::u1:
        return array.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    // need to create a view on array to cast it correctly
    case ov::element::f16:
    case ov::element::bf16:
        return array.view("int16").cast<py::array_t<int16_t, py::array::c_style | py::array::forcecast>>();
    default:
        OPENVINO_THROW("Tensor cannot be created as contiguous!");
    }
}

py::array array_from_tensor(ov::Tensor&& t, bool is_shared) {
    auto ov_type = t.get_element_type();
    auto dtype = Common::ov_type_to_dtype().at(ov_type);

    // Return the array as a view:
    if (is_shared) {
        if (ov_type.bitwidth() < Common::values::min_bitwidth) {
            return py::array(dtype, t.get_byte_size(), t.data(), py::cast(t));
        }
        return py::array(dtype, t.get_shape(), t.get_strides(), t.data(), py::cast(t));
    }
    // Return the array as a copy:
    if (ov_type.bitwidth() < Common::values::min_bitwidth) {
        return py::array(dtype, t.get_byte_size(), t.data());
    }
    return py::array(dtype, t.get_shape(), t.get_strides(), t.data());
}

};  // namespace array_helpers

template <>
ov::op::v0::Constant create_copied(py::array& array) {
    // Do not copy data from the array, only return empty tensor based on type.
    if (array.size() == 0) {
        return ov::op::v0::Constant(array_helpers::get_ov_type(array), array_helpers::get_shape(array));
    }
    // Convert to contiguous array if not already in C-style.
    if (!array_helpers::is_contiguous(array)) {
        array = array_helpers::as_contiguous(array, array_helpers::get_ov_type(array));
    }
    // Create actual Constant and a constructor is copying data.
    // If ndim is equal to 0, creates scalar Constant.
    // If size is equal to 0, creates empty Constant.
    return ov::op::v0::Constant(array_helpers::get_ov_type(array),
                                array_helpers::get_shape(array),
                                array.ndim() == 0 ? array.data() : array.data(0));
}

template <>
ov::op::v0::Constant create_copied(ov::Tensor& tensor) {
    // Create actual Constant and a constructor is copying data.
    return ov::op::v0::Constant(tensor.get_element_type(), tensor.get_shape(), const_cast<void*>(tensor.data()));
}

template <>
ov::op::v0::Constant create_shared(py::array& array) {
    // Check if passed array has C-style contiguous memory layout.
    // If memory is going to be shared it needs to be contiguous before passing to the constructor.
    // If ndim is equal to 0, creates scalar Constant.
    // If size is equal to 0, creates empty Constant.
    if (array_helpers::is_contiguous(array)) {
        auto memory = std::make_shared<ov::SharedBuffer<py::array>>(
            static_cast<char*>((array.ndim() == 0 || array.size() == 0) ? array.mutable_data() : array.mutable_data(0)),
            array.ndim() == 0 ? array.itemsize() : array.nbytes(),
            array);
        return ov::op::v0::Constant(array_helpers::get_ov_type(array), array_helpers::get_shape(array), memory);
    }
    // If passed array is not C-style, throw an error.
    OPENVINO_THROW("SHARED MEMORY MODE FOR THIS CONSTANT IS NOT APPLICABLE! Passed numpy array must be C contiguous.");
}

template <>
ov::op::v0::Constant create_shared(ov::Tensor& tensor) {
    return ov::op::v0::Constant(tensor);
}

template <>
ov::Tensor create_copied(py::array& array) {
    // Convert to contiguous array if not already in C-style.
    if (!array_helpers::is_contiguous(array)) {
        array = array_helpers::as_contiguous(array, array_helpers::get_ov_type(array));
    }
    // Create actual Tensor and copy data.
    auto tensor = ov::Tensor(array_helpers::get_ov_type(array), array_helpers::get_shape(array));
    // If ndim of py::array is 0, array is a numpy scalar. That results in size to be equal to 0.
    std::memcpy(tensor.data(),
                array.ndim() == 0 ? array.data() : array.data(0),
                array.ndim() == 0 ? array.itemsize() : array.nbytes());
    return tensor;
}

template <>
ov::Tensor create_shared(py::array& array) {
    // Check if passed array has C-style contiguous memory layout.
    // If memory is going to be shared it needs to be contiguous before passing to the constructor.
    if (array_helpers::is_contiguous(array)) {
        // If ndim of py::array is 0, array is a numpy scalar.
        return ov::Tensor(array_helpers::get_ov_type(array),
                          array_helpers::get_shape(array),
                          array.ndim() == 0 ? array.mutable_data() : array.mutable_data(0));
    }
    // If passed array is not C-style, throw an error.
    OPENVINO_THROW("SHARED MEMORY MODE FOR THIS TENSOR IS NOT APPLICABLE! Passed numpy array must be C contiguous.");
}

ov::Tensor tensor_from_pointer(py::array& array, const ov::Shape& shape, const ov::element::Type& type) {
    auto element_type = (type == ov::element::undefined) ? Common::dtype_to_ov_type().at(py::str(array.dtype())) : type;

    if (array_helpers::is_contiguous(array)) {
        return ov::Tensor(element_type, shape, const_cast<void*>(array.data(0)), {});
    }
    OPENVINO_THROW("SHARED MEMORY MODE FOR THIS TENSOR IS NOT APPLICABLE! Passed numpy array must be C contiguous.");
}

ov::Tensor tensor_from_pointer(py::array& array, const ov::Output<const ov::Node>& port) {
    auto array_type = array_helpers::get_ov_type(array);
    auto array_shape_size = ov::shape_size(array_helpers::get_shape(array));
    auto port_element_type = port.get_element_type();
    auto port_shape_size = ov::shape_size(port.get_partial_shape().is_dynamic() ? ov::Shape{0} : port.get_shape());

    if (array_helpers::is_contiguous(array)) {
        if (array_type != port_element_type) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Type of the array and the port are different. Data is going to be casted.",
                         1);
        }
        if (port.get_partial_shape().is_dynamic()) {
            return ov::Tensor(port, const_cast<void*>(array.data(0)));
        }
        if (port_shape_size > array_shape_size) {
            OPENVINO_THROW("Shape of the port exceeds shape of the array.");
        }
        if (port_shape_size < array_shape_size) {
            PyErr_WarnEx(PyExc_RuntimeWarning,
                         "Shape of the port is smaller than shape of the array. Passed data will be cropped.",
                         1);
        }
        return ov::Tensor(port, const_cast<void*>(array.data(0)));
    }

    OPENVINO_THROW("SHARED MEMORY MODE FOR THIS TENSOR IS NOT APPLICABLE! Passed numpy array must be C contiguous.");
}

ov::PartialShape partial_shape_from_list(const py::list& shape) {
    using value_type = ov::Dimension::value_type;
    ov::PartialShape pshape;
    for (py::handle dim : shape) {
        if (py::isinstance<py::int_>(dim)) {
            pshape.insert(pshape.end(), ov::Dimension(dim.cast<value_type>()));
        } else if (py::isinstance<py::str>(dim)) {
            pshape.insert(pshape.end(), ov::Dimension(dim.cast<std::string>()));
        } else if (py::isinstance<ov::Dimension>(dim)) {
            pshape.insert(pshape.end(), dim.cast<ov::Dimension>());
        } else if (py::isinstance<py::list>(dim) || py::isinstance<py::tuple>(dim)) {
            py::list bounded_dim = dim.cast<py::list>();
            if (bounded_dim.size() != 2) {
                throw py::type_error("Two elements are expected in tuple(lower, upper) for dynamic dimension, but " +
                                     std::to_string(bounded_dim.size()) + " elements were given.");
            }
            if (!(py::isinstance<py::int_>(bounded_dim[0]) && py::isinstance<py::int_>(bounded_dim[1]))) {
                throw py::type_error("Incorrect pair of types (" + std::string(py::str(bounded_dim[0].get_type())) +
                                     ", " + std::string(py::str(bounded_dim[1].get_type())) +
                                     ") for dynamic dimension, ints are expected.");
            }
            pshape.insert(pshape.end(),
                          ov::Dimension(bounded_dim[0].cast<value_type>(), bounded_dim[1].cast<value_type>()));
        } else {
            throw py::type_error("Incorrect type " + std::string(py::str(dim.get_type())) +
                                 " for dimension. Expected types are: "
                                 "int, str, openvino.runtime.Dimension, list/tuple with lower and upper values for "
                                 "dynamic dimension.");
        }
    }
    return pshape;
}

const ov::Tensor& cast_to_tensor(const py::handle& tensor) {
    return tensor.cast<const ov::Tensor&>();
}

void set_request_tensors(ov::InferRequest& request, const py::dict& inputs) {
    if (!inputs.empty()) {
        for (auto&& input : inputs) {
            // Cast second argument to tensor
            auto tensor = Common::cast_to_tensor(input.second);
            // Check if key is compatible, should be port/string/integer
            if (py::isinstance<ov::Output<const ov::Node>>(input.first)) {
                request.set_tensor(input.first.cast<ov::Output<const ov::Node>>(), tensor);
            } else if (py::isinstance<py::str>(input.first)) {
                request.set_tensor(input.first.cast<std::string>(), tensor);
            } else if (py::isinstance<py::int_>(input.first)) {
                request.set_input_tensor(input.first.cast<size_t>(), tensor);
            } else {
                throw py::type_error("Incompatible key type for tensor named: " + input.first.cast<std::string>());
            }
        }
    }
}

uint32_t get_optimal_number_of_requests(const ov::CompiledModel& actual) {
    try {
        auto supported_properties = actual.get_property(ov::supported_properties);
        OPENVINO_ASSERT(
            std::find(supported_properties.begin(), supported_properties.end(), ov::optimal_number_of_infer_requests) !=
                supported_properties.end(),
            "Can't load network: ",
            ov::optimal_number_of_infer_requests.name(),
            " is not supported!",
            " Please specify number of infer requests directly!");
        return actual.get_property(ov::optimal_number_of_infer_requests);
    } catch (const std::exception& ex) {
        OPENVINO_THROW("Can't load network: ", ex.what(), " Please specify number of infer requests directly!");
    }
}

py::dict outputs_to_dict(InferRequestWrapper& request, bool share_outputs) {
    py::dict res;
    for (const auto& out : request.m_outputs) {
        res[py::cast(out)] = array_helpers::array_from_tensor(request.m_request.get_tensor(out), share_outputs);
    }
    return res;
}

ov::pass::Serialize::Version convert_to_version(const std::string& version) {
    using Version = ov::pass::Serialize::Version;

    if (version == "UNSPECIFIED") {
        return Version::UNSPECIFIED;
    }
    if (version == "IR_V10") {
        return Version::IR_V10;
    }
    if (version == "IR_V11") {
        return Version::IR_V11;
    }
    OPENVINO_THROW("Invoked with wrong version argument: '",
                   version,
                   "'! The supported versions are: 'UNSPECIFIED'(default), 'IR_V10', 'IR_V11'.");
}

};  // namespace Common
