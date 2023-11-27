// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/constant.hpp"

#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "openvino/core/shape.hpp"
#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

template <typename T>
std::vector<size_t> _get_byte_strides(const ov::Shape& s) {
    std::vector<size_t> byte_strides;
    std::vector<size_t> element_strides = ov::row_major_strides(s);
    for (auto v : element_strides) {
        byte_strides.push_back(static_cast<size_t>(v) * sizeof(T));
    }
    return byte_strides;
}

std::vector<size_t> _get_strides(const ov::op::v0::Constant& self) {
    auto element_type = self.get_element_type();
    auto shape = self.get_shape();
    if (element_type == ov::element::boolean) {
        return _get_byte_strides<char>(shape);
    } else if (element_type == ov::element::f16 || element_type == ov::element::bf16) {
        // WA for bf16, returned as f16 array
        return _get_byte_strides<ov::float16>(shape);
    } else if (element_type == ov::element::f32) {
        return _get_byte_strides<float>(shape);
    } else if (element_type == ov::element::f64) {
        return _get_byte_strides<double>(shape);
    } else if (element_type == ov::element::i8 || element_type == ov::element::i4) {
        // WA for i4, returned as int8 array
        return _get_byte_strides<int8_t>(shape);
    } else if (element_type == ov::element::i16) {
        return _get_byte_strides<int16_t>(shape);
    } else if (element_type == ov::element::i32) {
        return _get_byte_strides<int32_t>(shape);
    } else if (element_type == ov::element::i64) {
        return _get_byte_strides<int64_t>(shape);
    } else if (element_type == ov::element::u8 || element_type == ov::element::u1 || element_type == ov::element::u4 ||
               element_type == ov::element::nf4) {
        // WA for u1, u4, nf4, all returned as packed uint8 arrays
        return _get_byte_strides<uint8_t>(shape);
    } else if (element_type == ov::element::u16) {
        return _get_byte_strides<uint16_t>(shape);
    } else if (element_type == ov::element::u32) {
        return _get_byte_strides<uint32_t>(shape);
    } else if (element_type == ov::element::u64) {
        return _get_byte_strides<uint64_t>(shape);
    } else {
        throw std::runtime_error("Unsupported data type!");
    }
}

// TODO: Remove in future and re-use `get_data`
template <typename T>
py::buffer_info _get_buffer_info(const ov::op::v0::Constant& c) {
    ov::Shape shape = c.get_shape();
    return py::buffer_info(const_cast<void*>(c.get_data_ptr()),              /* Pointer to buffer */
                           static_cast<size_t>(c.get_element_type().size()), /* Size of one scalar */
                           py::format_descriptor<T>::format(),               /* Python struct-style format descriptor */
                           static_cast<size_t>(shape.size()),                /* Number of dimensions */
                           std::vector<size_t>{shape.begin(), shape.end()},  /* Buffer dimensions */
                           _get_byte_strides<T>(shape)                       /* Strides (in bytes) for each index */
    );
}

// TODO: Remove in future and re-use `get_data`
template <>
py::buffer_info _get_buffer_info<ov::float16>(const ov::op::v0::Constant& c) {
    ov::Shape shape = c.get_shape();
    return py::buffer_info(const_cast<void*>(c.get_data_ptr()),              /* Pointer to buffer */
                           static_cast<size_t>(c.get_element_type().size()), /* Size of one scalar */
                           std::string(1, 'H'),                              /* Python struct-style format descriptor */
                           static_cast<size_t>(shape.size()),                /* Number of dimensions */
                           std::vector<size_t>{shape.begin(), shape.end()},  /* Buffer dimensions */
                           _get_byte_strides<ov::float16>(shape)             /* Strides (in bytes) for each index */
    );
}

template <typename T>
py::array _cast_vector(const ov::op::v0::Constant& self) {
    auto vec = self.cast_vector<T>();
    return py::array(vec.size(), vec.data());
}

template <>
py::array _cast_vector<ov::float16>(const ov::op::v0::Constant& self) {
    auto vec = self.cast_vector<ov::float16>();
    return py::array(py::dtype("float16"), vec.size(), vec.data());
}

void regclass_graph_op_Constant(py::module m) {
    py::class_<ov::op::v0::Constant, std::shared_ptr<ov::op::v0::Constant>, ov::Node> constant(m,
                                                                                               "Constant",
                                                                                               py::buffer_protocol());
    constant.doc() = "openvino.runtime.op.Constant wraps ov::op::v0::Constant";
    // Numpy-based constructor
    constant.def(py::init([](py::array& array, bool shared_memory) {
                     return Common::object_from_data<ov::op::v0::Constant>(array, shared_memory);
                 }),
                 py::arg("array"),
                 py::arg("shared_memory") = false);
    // Tensor-based constructors
    constant.def(py::init([](ov::Tensor& tensor, bool shared_memory) {
                     return Common::object_from_data<ov::op::v0::Constant>(tensor, shared_memory);
                 }),
                 py::arg("tensor"),
                 py::arg("shared_memory") = false);
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<char>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<ov::float16>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<float>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<double>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<int8_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<int16_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<int32_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<int64_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<uint8_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<uint16_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<uint32_t>&>());
    constant.def(py::init<const ov::element::Type&, const ov::Shape&, const std::vector<uint64_t>&>());

    constant.def("get_value_strings", &ov::op::v0::Constant::get_value_strings);

    constant.def("get_byte_size", &ov::op::v0::Constant::get_byte_size);

    constant.def("get_vector", [](const ov::op::v0::Constant& self) {
        auto element_type = self.get_element_type();
        if (element_type == ov::element::boolean) {
            return _cast_vector<char>(self);
        } else if (element_type == ov::element::f16) {
            return _cast_vector<ov::float16>(self);
        } else if (element_type == ov::element::f32) {
            return _cast_vector<float>(self);
        } else if (element_type == ov::element::f64) {
            return _cast_vector<double>(self);
        } else if (element_type == ov::element::i8) {
            return _cast_vector<int8_t>(self);
        } else if (element_type == ov::element::i16) {
            return _cast_vector<int16_t>(self);
        } else if (element_type == ov::element::i32) {
            return _cast_vector<int32_t>(self);
        } else if (element_type == ov::element::i64) {
            return _cast_vector<int64_t>(self);
        } else if (element_type == ov::element::u8 || element_type == ov::element::u1) {
            return _cast_vector<uint8_t>(self);
        } else if (element_type == ov::element::u16) {
            return _cast_vector<uint16_t>(self);
        } else if (element_type == ov::element::u32) {
            return _cast_vector<uint32_t>(self);
        } else if (element_type == ov::element::u64) {
            return _cast_vector<uint64_t>(self);
        } else {
            throw std::runtime_error("Unsupported data type!");
        }
    });

    // TODO: Remove in future and re-use `get_data`
    // Provide buffer access
    constant.def_buffer([](const ov::op::v0::Constant& self) -> py::buffer_info {
        auto element_type = self.get_element_type();
        if (element_type == ov::element::boolean) {
            return _get_buffer_info<char>(self);
        } else if (element_type == ov::element::f16) {
            return _get_buffer_info<ov::float16>(self);
        } else if (element_type == ov::element::f32) {
            return _get_buffer_info<float>(self);
        } else if (element_type == ov::element::f64) {
            return _get_buffer_info<double>(self);
        } else if (element_type == ov::element::i8) {
            return _get_buffer_info<int8_t>(self);
        } else if (element_type == ov::element::i16) {
            return _get_buffer_info<int16_t>(self);
        } else if (element_type == ov::element::i32) {
            return _get_buffer_info<int32_t>(self);
        } else if (element_type == ov::element::i64) {
            return _get_buffer_info<int64_t>(self);
        } else if (element_type == ov::element::u8 || element_type == ov::element::u1) {
            return _get_buffer_info<uint8_t>(self);
        } else if (element_type == ov::element::u16) {
            return _get_buffer_info<uint16_t>(self);
        } else if (element_type == ov::element::u32) {
            return _get_buffer_info<uint32_t>(self);
        } else if (element_type == ov::element::u64) {
            return _get_buffer_info<uint64_t>(self);
        } else {
            throw std::runtime_error("Unsupported data type!");
        }
    });

    constant.def(
        "get_data",
        [](ov::op::v0::Constant& self) {
            auto ov_type = self.get_element_type();
            auto dtype = Common::ov_type_to_dtype().at(ov_type);
            if (ov_type.bitwidth() < Common::values::min_bitwidth) {
                return py::array(dtype, self.get_byte_size(), self.get_data_ptr());
            }
            return py::array(dtype, self.get_shape(), _get_strides(self), self.get_data_ptr());
        },
        R"(
            Access to Constant's data - creates a copy of data.

            Returns numpy array with corresponding shape and dtype.
            For Constants with openvino specific element type, such as u1,
            it returns linear array, with uint8 / int8 numpy dtype.

            :rtype: numpy.array
        )");

    constant.def_property_readonly(
        "data",
        [](ov::op::v0::Constant& self) {
            auto ov_type = self.get_element_type();
            auto dtype = Common::ov_type_to_dtype().at(ov_type);
            if (ov_type.bitwidth() < Common::values::min_bitwidth) {
                return py::array(dtype, self.get_byte_size(), self.get_data_ptr(), py::cast(self));
            }
            return py::array(dtype, self.get_shape(), _get_strides(self), self.get_data_ptr(), py::cast(self));
        },
        R"(
            Access to Constant's data - creates a view of data.

            Returns numpy array with corresponding shape and dtype.
            For Constants with openvino specific element type, such as u1,
            it returns linear array, with uint8 / int8 numpy dtype.

            Note: this access method reflects shared memory if it was applied during initialization.

            :rtype: numpy.array
        )");

    constant.def("__repr__", [](const ov::op::v0::Constant& self) {
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + Common::get_class_name(self) + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });
}
