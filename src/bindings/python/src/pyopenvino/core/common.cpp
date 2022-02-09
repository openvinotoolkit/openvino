// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

#include <unordered_map>

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

ov::Tensor tensor_from_numpy(py::array& array, bool shared_memory) {
    // Check if passed array has C-style contiguous memory layout.
    bool is_contiguous = C_CONTIGUOUS == (array.flags() & C_CONTIGUOUS);
    auto type = Common::dtype_to_ov_type().at(py::str(array.dtype()));
    std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());

    // If memory is going to be shared it needs to be contiguous before
    // passing to the constructor. This case should be handled by advanced
    // users on their side of the code.
    if (shared_memory) {
        if (is_contiguous) {
            std::vector<size_t> strides(array.strides(), array.strides() + array.ndim());
            return ov::Tensor(type, shape, const_cast<void*>(array.data(0)), strides);
        } else {
            throw ov::Exception("Tensor with shared memory must be C contiguous!");
        }
    }
    // Convert to contiguous array if not already C-style.
    if (!is_contiguous) {
        array = Common::as_contiguous(array, type);
    }
    // Create actual Tensor and copy data.
    auto tensor = ov::Tensor(type, shape);
    // If ndim of py::array is 0, array is a numpy scalar. That results in size to be equal to 0.
    // To gain access to actual raw/low-level data, it is needed to use buffer protocol.
    py::buffer_info buf = array.request();
    std::memcpy(tensor.data(), buf.ptr, buf.ndim == 0 ? buf.itemsize : buf.itemsize * buf.size);
    return tensor;
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
        throw ov::Exception("Tensor cannot be created as contiguous!");
        break;
    }
}

const ov::Tensor& cast_to_tensor(const py::handle& tensor) {
    return tensor.cast<const ov::Tensor&>();
}

const Containers::TensorNameMap cast_to_tensor_name_map(const py::dict& inputs) {
    Containers::TensorNameMap result_map;
    for (auto&& input : inputs) {
        std::string name;
        if (py::isinstance<py::str>(input.first)) {
            name = input.first.cast<std::string>();
        } else {
            throw py::type_error("incompatible function arguments!");
        }
        if (py::isinstance<ov::Tensor>(input.second)) {
            auto tensor = Common::cast_to_tensor(input.second);
            result_map[name] = tensor;
        } else {
            throw ov::Exception("Unable to cast tensor " + name + "!");
        }
    }
    return result_map;
}

const Containers::TensorIndexMap cast_to_tensor_index_map(const py::dict& inputs) {
    Containers::TensorIndexMap result_map;
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
            throw ov::Exception("Unable to cast tensor " + std::to_string(idx) + "!");
        }
    }
    return result_map;
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

PyAny from_ov_any(const ov::Any& any) {
    // Check for py::object
    if (any.is<py::object>()) {
        return any.as<py::object>();
    }
    // Check for std::string
    else if (any.is<std::string>()) {
        return PyUnicode_FromString(any.as<std::string>().c_str());
    }
    // Check for int
    else if (any.is<int>()) {
        auto val = any.as<int>();
        return PyLong_FromLong((long)val);
    } else if (any.is<int64_t>()) {
        auto val = any.as<int64_t>();
        return PyLong_FromLong((long)val);
    }
    // Check for unsinged int
    else if (any.is<unsigned int>()) {
        auto val = any.as<unsigned int>();
        return PyLong_FromLong((unsigned long)val);
    }
    // Check for float
    else if (any.is<float>()) {
        auto val = any.as<float>();
        return PyFloat_FromDouble((double)val);
    } else if (any.is<double>()) {
        auto val = any.as<double>();
        return PyFloat_FromDouble(val);
    }
    // Check for bool
    else if (any.is<bool>()) {
        auto val = any.as<bool>();
        return val ? Py_True : Py_False;
    }
    // Check for std::vector<std::string>
    else if (any.is<std::vector<std::string>>()) {
        auto val = any.as<std::vector<std::string>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyObject* str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
    // Check for std::vector<int>
    else if (any.is<std::vector<int>>()) {
        auto val = any.as<std::vector<int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
    // Check for std::vector<int64_t>
    else if (any.is<std::vector<int64_t>>()) {
        auto val = any.as<std::vector<int64_t>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
    // Check for std::vector<unsigned int>
    else if (any.is<std::vector<unsigned int>>()) {
        auto val = any.as<std::vector<unsigned int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
    // Check for std::vector<float>
    else if (any.is<std::vector<float>>()) {
        auto val = any.as<std::vector<float>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyFloat_FromDouble((double)it));
        }
        return list;
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int>>()) {
        auto val = any.as<std::tuple<unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return tuple;
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (any.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto val = any.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return tuple;
    }
    // Check for std::map<std::string, std::string>
    else if (any.is<std::map<std::string, std::string>>()) {
        auto val = any.as<std::map<std::string, std::string>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
    // Check for std::map<std::string, int>
    else if (any.is<std::map<std::string, int>>()) {
        auto val = any.as<std::map<std::string, int>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return dict;
    }
    // Check for std::vector<ov::PropertyName>
    else if (any.is<std::vector<ov::PropertyName>>()) {
        auto val = any.as<std::vector<ov::PropertyName>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            std::string property_name = it;
            std::string mutability = it.is_mutable() ? "RW" : "RO";
            PyDict_SetItemString(dict, property_name.c_str(), PyUnicode_FromString(mutability.c_str()));
        }
        return dict;
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject*)NULL;
    }
}

uint32_t get_optimal_number_of_requests(const ov::CompiledModel& actual) {
    try {
        auto supported_properties = actual.get_property(ov::supported_properties);
        if (std::find(supported_properties.begin(), supported_properties.end(), ov::optimal_number_of_infer_requests) !=
            supported_properties.end()) {
            return actual.get_property(ov::optimal_number_of_infer_requests);
        } else {
            IE_THROW() << "Can't load network: " << ov::optimal_number_of_infer_requests.name() << " is not supported!"
                       << " Please specify number of infer requests directly!";
        }
    } catch (const std::exception& ex) {
        IE_THROW() << "Can't load network: " << ex.what() << " Please specify number of infer requests directly!";
    }
}

py::dict outputs_to_dict(const std::vector<ov::Output<const ov::Node>>& outputs, ov::InferRequest& request) {
    py::dict res;
    for (const auto& out : outputs) {
        ov::Tensor t{request.get_tensor(out)};
        switch (t.get_element_type()) {
        case ov::element::Type_t::i8: {
            res[py::cast(out)] = py::array_t<int8_t>(t.get_shape(), t.data<int8_t>());
            break;
        }
        case ov::element::Type_t::i16: {
            res[py::cast(out)] = py::array_t<int16_t>(t.get_shape(), t.data<int16_t>());
            break;
        }
        case ov::element::Type_t::i32: {
            res[py::cast(out)] = py::array_t<int32_t>(t.get_shape(), t.data<int32_t>());
            break;
        }
        case ov::element::Type_t::i64: {
            res[py::cast(out)] = py::array_t<int64_t>(t.get_shape(), t.data<int64_t>());
            break;
        }
        case ov::element::Type_t::u8: {
            res[py::cast(out)] = py::array_t<uint8_t>(t.get_shape(), t.data<uint8_t>());
            break;
        }
        case ov::element::Type_t::u16: {
            res[py::cast(out)] = py::array_t<uint16_t>(t.get_shape(), t.data<uint16_t>());
            break;
        }
        case ov::element::Type_t::u32: {
            res[py::cast(out)] = py::array_t<uint32_t>(t.get_shape(), t.data<uint32_t>());
            break;
        }
        case ov::element::Type_t::u64: {
            res[py::cast(out)] = py::array_t<uint64_t>(t.get_shape(), t.data<uint64_t>());
            break;
        }
        case ov::element::Type_t::bf16: {
            res[py::cast(out)] = py::array(py::dtype("float16"), t.get_shape(), t.data<ov::bfloat16>());
            break;
        }
        case ov::element::Type_t::f16: {
            res[py::cast(out)] = py::array(py::dtype("float16"), t.get_shape(), t.data<ov::float16>());
            break;
        }
        case ov::element::Type_t::f32: {
            res[py::cast(out)] = py::array_t<float>(t.get_shape(), t.data<float>());
            break;
        }
        case ov::element::Type_t::f64: {
            res[py::cast(out)] = py::array_t<double>(t.get_shape(), t.data<double>());
            break;
        }
        case ov::element::Type_t::boolean: {
            res[py::cast(out)] = py::array_t<bool>(t.get_shape(), t.data<bool>());
            break;
        }
        default: {
            break;
        }
        }
    }
    return res;
}

};  // namespace Common
