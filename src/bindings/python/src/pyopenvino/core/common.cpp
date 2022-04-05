// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"

#include <unordered_map>

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

ov::Tensor tensor_from_pointer(py::array& array, const ov::Shape& shape, const ov::element::Type& type) {
    bool is_contiguous = C_CONTIGUOUS == (array.flags() & C_CONTIGUOUS);
    auto element_type = (type == ov::element::undefined) ? Common::dtype_to_ov_type().at(py::str(array.dtype())) : type;

    if (is_contiguous) {
        return ov::Tensor(element_type, shape, const_cast<void*>(array.data(0)), {});
    } else {
        throw ov::Exception("Tensor with shared memory must be C contiguous!");
    }
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

ov::PartialShape partial_shape_from_list(const py::list& shape) {
    using value_type = ov::Dimension::value_type;
    ov::PartialShape pshape;
    for (py::handle dim : shape) {
        if (py::isinstance<py::int_>(dim)) {
            pshape.insert(pshape.end(), ov::Dimension(dim.cast<value_type>()));
        } else if (py::isinstance<py::str>(dim)) {
            pshape.insert(pshape.end(), Common::dimension_from_str(dim.cast<std::string>()));
        } else if (py::isinstance<ov::Dimension>(dim)) {
            pshape.insert(pshape.end(), dim.cast<ov::Dimension>());
        } else if (py::isinstance<py::list>(dim) || py::isinstance<py::tuple>(dim)) {
            py::list bounded_dim = dim.cast<py::list>();
            if (bounded_dim.size() != 2) {
                throw py::type_error("Two elements are expected in tuple(lower, upper) for dynamic dimension, but " +
                                     std::to_string(bounded_dim.size()) + " elements were given.");
            }
            if (!(py::isinstance<py::int_>(bounded_dim[0]) && py::isinstance<py::int_>(bounded_dim[1]))) {
                throw py::type_error("Incorrect pair of types (" + std::string(bounded_dim[0].get_type().str()) + ", " +
                                     std::string(bounded_dim[1].get_type().str()) +
                                     ") for dynamic dimension, ints are expected.");
            }
            pshape.insert(pshape.end(),
                          ov::Dimension(bounded_dim[0].cast<value_type>(), bounded_dim[1].cast<value_type>()));
        } else {
            throw py::type_error("Incorrect type " + std::string(dim.get_type().str()) +
                                 " for dimension. Expected types are: "
                                 "int, str, openvino.runtime.Dimension, list/tuple with lower and upper values for "
                                 "dynamic dimension.");
        }
    }
    return pshape;
}

bool check_all_digits(const std::string& value) {
    auto val = ov::util::trim(value);
    for (const auto& c : val) {
        if (!std::isdigit(c) || c == '-') {
            return false;
        }
    }
    return true;
}

template <class T>
T stringToType(const std::string& valStr) {
    T ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}

ov::Dimension dimension_from_str(const std::string& value) {
    using value_type = ov::Dimension::value_type;
    auto val = ov::util::trim(value);
    if (val == "?" || val == "-1") {
        return {-1};
    }
    if (val.find("..") == std::string::npos) {
        OPENVINO_ASSERT(Common::check_all_digits(val), "Cannot parse dimension: \"", val, "\"");
        return {Common::stringToType<value_type>(val)};
    }

    std::string min_value_str = val.substr(0, val.find(".."));
    OPENVINO_ASSERT(Common::check_all_digits(min_value_str), "Cannot parse min bound: \"", min_value_str, "\"");

    value_type min_value;
    if (min_value_str.empty()) {
        min_value = 0;
    } else {
        min_value = Common::stringToType<value_type>(min_value_str);
    }

    std::string max_value_str = val.substr(val.find("..") + 2);
    value_type max_value;
    if (max_value_str.empty()) {
        max_value = -1;
    } else {
        max_value = Common::stringToType<value_type>(max_value_str);
    }

    OPENVINO_ASSERT(Common::check_all_digits(max_value_str), "Cannot parse max bound: \"", max_value_str, "\"");

    return {min_value, max_value};
}

ov::PartialShape partial_shape_from_str(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val == "...") {
        return ov::PartialShape::dynamic();
    }
    ov::PartialShape res;
    std::stringstream ss(val);
    std::string field;
    while (getline(ss, field, ',')) {
        OPENVINO_ASSERT(!field.empty(), "Cannot get vector of dimensions! \"", val, "\" is incorrect");
        res.insert(res.end(), Common::dimension_from_str(field));
    }
    return res;
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
    throw ov::Exception("Invoked with wrong version argument: '" + version +
                        "'! The supported versions are: 'UNSPECIFIED'(default), 'IR_V10', 'IR_V11'.");
}

};  // namespace Common
