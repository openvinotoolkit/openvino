// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>

#include <map>

#include "openvino/runtime/tensor.hpp"

namespace py = pybind11;
using namespace ov::runtime;

std::map<ov::element::Type, pybind11::dtype> ov_type_to_dtype = {
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

std::map<py::str, ov::element::Type> dtype_to_ov_type = {
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


Tensor create_Tensor(const ov::element::Type type, py::array& array) {
    std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
    std::vector<size_t> strides(array.strides(), array.strides() + array.ndim());
    return Tensor(type, shape, (void*)array.data(), strides);
}

void regclass_Tensor(py::module m) {
    py::class_<Tensor, std::shared_ptr<Tensor>> cls(m, "Tensor");

    cls.def(py::init());
    cls.def(py::init<const ov::element::Type, const ov::Shape>());
    cls.def(py::init([](py::array& array) {
        return create_Tensor(dtype_to_ov_type[py::str(array.dtype())], array);
    }));

    cls.def_property_readonly("element_type", &Tensor::get_element_type);
    cls.def_property("shape", &Tensor::get_shape, &Tensor::set_shape);

    cls.def_property_readonly("data", [](Tensor& self) {
        ov::element::Type dtype = self.get_element_type();
        return py::array(ov_type_to_dtype[dtype], self.get_shape(), self.data(), py::cast(self));
    });
}
