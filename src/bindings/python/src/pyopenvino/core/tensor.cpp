// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_Tensor(py::module m) {
    py::class_<ov::runtime::Tensor, std::shared_ptr<ov::runtime::Tensor>> cls(m, "Tensor");

    cls.def(py::init([](py::array& array, bool shared_memory) {
                return Common::tensor_from_numpy(array, shared_memory);
            }),
            py::arg("array"),
            py::arg("shared_memory") = false);

    cls.def(py::init<const ov::element::Type, const ov::Shape>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init<const ov::element::Type, const std::vector<size_t>>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, std::vector<size_t>& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, std::vector<size_t>& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))),
                                           shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, const ov::Shape& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, const ov::Shape& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))),
                                           shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init<ov::runtime::Tensor, ov::Coordinate, ov::Coordinate>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def(py::init<ov::runtime::Tensor, std::vector<size_t>, std::vector<size_t>>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def_property_readonly("element_type", &ov::runtime::Tensor::get_element_type);

    cls.def_property_readonly("size", &ov::runtime::Tensor::get_size);

    cls.def_property_readonly("byte_size", &ov::runtime::Tensor::get_byte_size);

    cls.def_property_readonly("data", [](ov::runtime::Tensor& self) {
        return py::array(Common::ov_type_to_dtype().at(self.get_element_type()),
                         self.get_shape(),
                         self.get_strides(),
                         self.data(),
                         py::cast(self));
    });

    cls.def_property("shape", &ov::runtime::Tensor::get_shape, &ov::runtime::Tensor::set_shape);

    cls.def_property("shape",
                     &ov::runtime::Tensor::get_shape,
                     [](ov::runtime::Tensor& self, std::vector<size_t>& shape) {
                         self.set_shape(shape);
                     });
}
