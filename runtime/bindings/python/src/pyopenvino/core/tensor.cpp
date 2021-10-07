// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;
using namespace ov::runtime;

void regclass_Tensor(py::module m) {
    py::class_<Tensor, std::shared_ptr<Tensor>> cls(m, "Tensor");

    cls.def(py::init<const ov::element::Type, const ov::Shape>());
    cls.def(py::init([](py::array& array) {
        std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
        std::vector<size_t> strides(array.strides(), array.strides() + array.ndim());
        ov::element::Type ov_type = Common::dtype_to_ov_type[py::str(array.dtype())];
        return Tensor(ov_type, shape, (void*)array.data(), strides);
    }));
    cls.def(py::init([](py::dtype np_dtype, std::vector<size_t> shape) {
        return Tensor(Common::dtype_to_ov_type[py::str(np_dtype)], shape);
    }));
    cls.def(py::init([](py::object np_literal, std::vector<size_t> shape) {
        return Tensor(Common::dtype_to_ov_type[py::str(py::dtype::from_args(np_literal))], shape);
    }));

    cls.def_property_readonly("element_type", &Tensor::get_element_type);
    cls.def_property_readonly("data", [](Tensor& self) {
        ov::element::Type ov_type = self.get_element_type();
        return py::array(Common::ov_type_to_dtype[ov_type], self.get_shape(), self.data(), py::cast(self));
    });
    cls.def_property("shape", &Tensor::get_shape, &Tensor::set_shape);
    cls.def_property("shape", &Tensor::get_shape, [](Tensor& self, std::vector<size_t> shape) {
        self.set_shape(shape);
    });
}
